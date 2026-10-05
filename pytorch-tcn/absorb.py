# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import qonnx.core.data_layout as DataLayout
import warnings
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name


def _as_channel_vector(arr):
    """
    把可吸收的 broadcast 常量统一变成 1D channel vector。

    支持：
      - scalar:           [] / [1] / [1,1,...]
      - 1D vector:        [C]
      - channel-broadcast [1,C,1] / [1,C,1,1]

    返回：
      - scalar: numpy scalar
      - vector: shape [C]
      - 不支持则返回 None
    """
    arr = np.asarray(arr)

    # scalar
    if arr.ndim == 0:
        return arr

    # all ones => scalar
    if all(x == 1 for x in arr.shape):
        return arr.flatten()[0]

    # already 1D
    if arr.ndim == 1:
        return arr

    # channel-broadcast form: only axis 1 is non-singleton
    nz_axes = [i for i, d in enumerate(arr.shape) if d != 1]
    if nz_axes == [1]:
        return arr.reshape(arr.shape[1])

    return None

def _get_const_and_dyn_input(model, node):
    """
    对于二输入节点(Add/Mul/Sub/Div)，返回:
      const_name, const_value, dyn_name
    要求恰好一个输入是 initializer，另一个是动态张量。
    否则返回 (None, None, None)
    """
    if len(node.input) != 2:
        return None, None, None

    a_name = node.input[0]
    b_name = node.input[1]

    a_init = model.get_initializer(a_name)
    b_init = model.get_initializer(b_name)

    if a_init is not None and b_init is None:
        return a_name, a_init, b_name
    elif b_init is not None and a_init is None:
        return b_name, b_init, a_name
    else:
        return None, None, None


def _get_single_consumer(model, tensor_name):
    consumers = model.find_consumers(tensor_name)
    if consumers is None or len(consumers) != 1:
        return None
    return consumers[0]

class AbsorbSignBiasIntoMultiThreshold(Transformation):
    """Absorb scalar/channelwise bias originating from signed int export back into
    MultiThreshold and re-evaluate the output datatype."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if (
                n.op_type == "MultiThreshold"
                and not model.is_fork_node(n)
                and not model.is_join_node(n)
            ):
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "Add":
                    mt_node = n
                    add_node = consumer
                    threshold_name = mt_node.input[1]
                    add_weight_name = add_node.input[1]

                    T = model.get_initializer(threshold_name)
                    A = model.get_initializer(add_weight_name)

                    if (A is None) or (T is None):
                        warnings.warn("Threshold or add bias not constant, skipping")
                        continue

                    end_name = add_node.output[0]

                    Avec = _as_channel_vector(A)
                    if Avec is None:
                        continue

                    mt_inst = getCustomOp(mt_node)

                    # 这里只吸收 scalar；如果是 per-channel bias，直接跳过
                    # 因为 out_bias 是节点属性，不是 per-channel tensor
                    if np.isscalar(Avec) or np.ndim(Avec) == 0:
                        bias = float(np.asarray(Avec))
                    else:
                        continue

                    bias += mt_inst.get_nodeattr("out_bias")
                    mt_inst.set_nodeattr("out_bias", bias)
                    graph_modified = True

                    steps = T.shape[-1]
                    new_min = bias
                    new_max = steps + bias
                    odt = DataType.get_smallest_possible(steps).name.replace("UINT", "INT")
                    odt = DataType[odt]
                    assert odt.allowed(new_max) and odt.allowed(
                        new_min
                    ), """Could
                    not compute new MultiThreshold DataType (min = %d max = %d)""" % (
                        new_min,
                        new_max,
                    )
                    mt_inst.set_nodeattr("out_dtype", odt.name)

                    graph.node.remove(add_node)
                    mt_node.output[0] = end_name
                    model.set_tensor_datatype(end_name, odt)

        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class AbsorbAddIntoMultiThreshold(Transformation):
    """Absorb preceding Add ops into MultiThreshold by updating the threshold
    values. Supports scalar / 1D / channel-broadcast add vectors."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        for n in list(graph.node):
            if n.op_type != "Add":
                continue

            consumer = _get_single_consumer(model, n.output[0])
            if consumer is None or consumer.op_type != "MultiThreshold":
                continue

            const_name, A, start_name = _get_const_and_dyn_input(model, n)
            if A is None:
                continue

            threshold_name = consumer.input[1]
            T = model.get_initializer(threshold_name)
            assert T is not None, "Initializer for thresholds is not set."

            Avec = _as_channel_vector(A)
            if Avec is None:
                continue

            if np.isscalar(Avec) or np.ndim(Avec) == 0:
                Tnew = T - float(np.asarray(Avec))
            else:
                if T.shape[0] != len(Avec):
                    continue
                Tnew = T - np.asarray(Avec).reshape(-1, 1)

            model.set_initializer(threshold_name, Tnew)
            consumer.input[0] = start_name
            graph.node.remove(n)
            graph_modified = True

        return (model, graph_modified)
    


class AbsorbMulIntoMultiThreshold(Transformation):
    """Absorb preceding Mul ops into MultiThreshold by updating the threshold
    values. Supports positive scalar / 1D / channel-broadcast mul vectors."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        for n in list(graph.node):
            if n.op_type != "Mul":
                continue

            consumer = _get_single_consumer(model, n.output[0])
            if consumer is None or consumer.op_type != "MultiThreshold":
                continue

            const_name, A, start_name = _get_const_and_dyn_input(model, n)
            if A is None:
                continue

            Avec = _as_channel_vector(A)
            if Avec is None:
                continue

            if np.any(np.asarray(Avec) <= 0):
                continue

            threshold_name = consumer.input[1]
            T = model.get_initializer(threshold_name)
            assert T is not None, "Initializer for thresholds is not set."

            if np.isscalar(Avec) or np.ndim(Avec) == 0:
                Tnew = T / float(np.asarray(Avec))
            else:
                if T.shape[0] != len(Avec):
                    continue
                Tnew = T / np.asarray(Avec).reshape(-1, 1)

            model.set_initializer(threshold_name, Tnew)
            consumer.input[0] = start_name
            graph.node.remove(n)
            graph_modified = True

        return (model, graph_modified)


class FactorOutMulSignMagnitude(Transformation):
    """Split multiply-by-constant nodes into two multiply-by-constant nodes,
    where the first node is a bipolar vector (of signs) and the second is a
    vector of magnitudes."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Mul":
                mul_weight_name = n.input[1]
                A = model.get_initializer(mul_weight_name)
                assert A is not None, "Initializer for mul weights is not set."
                is_scalar = np.prod(A.shape) == 1
                actual_ndims = len(tuple(filter(lambda x: x > 1, A.shape)))
                is_1d = actual_ndims == 1
                is_not_bipolar = model.get_tensor_datatype(mul_weight_name) != DataType["BIPOLAR"]
                is_signed = (A < 0).any()
                if is_signed and (is_scalar or is_1d) and is_not_bipolar:
                    start_name = n.input[0]
                    in_shape = model.get_tensor_shape(start_name)
                    middle_name = model.make_new_valueinfo_name()
                    model.set_tensor_shape(middle_name, in_shape)
                    sign_mul_param_name = model.make_new_valueinfo_name()
                    # create new mul node with sign(A) as the operand
                    sgn = np.sign(A)
                    model.set_initializer(sign_mul_param_name, sgn)
                    model.set_tensor_datatype(sign_mul_param_name, DataType["BIPOLAR"])
                    # replace original mul weight by magnitudes
                    model.set_initializer(mul_weight_name, np.abs(A))
                    new_mul = oh.make_node("Mul", [start_name, sign_mul_param_name], [middle_name])
                    n.input[0] = middle_name
                    graph.node.insert(node_ind - 1, new_mul)
                    graph_modified = True
        return (model, graph_modified)


class Absorb1BitMulIntoMatMul(Transformation):
    """Absorb bipolar or binary multiplications into the preciding matrix
    multiply."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "MatMul":
                matmul_weight_name = n.input[1]
                W = model.get_initializer(matmul_weight_name)
                Wdt = model.get_tensor_datatype(matmul_weight_name)
                assert W is not None, "Initializer for matmul weights is not set."
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "Mul":
                    mul_weight_name = consumer.input[1]
                    A = model.get_initializer(mul_weight_name)
                    assert A is not None, "Initializer for mul weights is not set."
                    is_1bit = model.get_tensor_datatype(mul_weight_name).bitwidth() == 1
                    if is_1bit:
                        Wnew = A * W
                        assert (
                            Wnew.shape == W.shape
                        ), """Shape of new weights is not
                        the same as the shape of the weight matrix before."""
                        check_fxn = np.vectorize(lambda x: Wdt.allowed(x))
                        # only absorb if permitted by W datatype
                        if check_fxn(Wnew).all():
                            model.set_initializer(matmul_weight_name, Wnew)
                            n.output[0] = consumer.output[0]
                            graph.node.remove(consumer)
                            graph_modified = True
        return (model, graph_modified)


class Absorb1BitMulIntoConv(Transformation):
    """Absorb bipolar or binary multiplications into the preciding convolution."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Conv":
                conv_weight_name = n.input[1]
                W = model.get_initializer(conv_weight_name)
                Wdt = model.get_tensor_datatype(conv_weight_name)
                assert W is not None, "Initializer for conv weights is not set."
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "Mul":
                    mul_weight_name = consumer.input[1]
                    A = model.get_initializer(mul_weight_name)
                    assert A is not None, "Initializer for mul weights is not set."
                    is_1bit = model.get_tensor_datatype(mul_weight_name).bitwidth() == 1
                    is_scalar = np.prod(A.shape) == 1
                    actual_ndims = len(tuple(filter(lambda x: x > 1, A.shape)))
                    is_1d = actual_ndims == 1
                    if is_1bit and (is_1d or is_scalar):
                        # move the mul to the OFM position, since the mul is
                        # applied on the outputs channelwise or as scalar
                        Wnew = A.reshape(-1, 1, 1, 1) * W
                        assert (
                            Wnew.shape == W.shape
                        ), """Shape of new weights is not
                        the same as the shape of the conv weights before."""
                        check_fxn = np.vectorize(lambda x: Wdt.allowed(x))
                        # only absorb if permitted by W datatype
                        if check_fxn(Wnew).all():
                            model.set_initializer(conv_weight_name, Wnew)
                            n.output[0] = consumer.output[0]
                            graph.node.remove(consumer)
                            graph_modified = True
        return (model, graph_modified)


class AbsorbTransposeIntoMultiThreshold(Transformation):
    """For (NCHWTranspose -> MultiThreshold) move Transpose past MultiThreshold
    and set its data_layout mode to NHWC."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        nodes = [n for n in model.graph.node]
        for n in nodes:
            node_ind += 1
            if n.op_type == "Transpose" and not model.is_fork_node(n):
                perms = list(get_by_name(n.attribute, "perm").ints)
                if perms == [0, 3, 1, 2]:
                    mt_cand = model.find_consumer(n.output[0])
                    if (
                        mt_cand is not None
                        and mt_cand.op_type == "MultiThreshold"
                        # and not model.is_fork_node(mt_cand)
                    ):
                        mt_cand_orig_output = mt_cand.output[0]
                        mt = getCustomOp(mt_cand)
                        mt.set_nodeattr("data_layout", "NHWC")
                        # Rewire input of MultiThreshold node
                        mt_cand.input[0] = n.input[0]
                        # Make new intermediate tensor
                        intermediate_tensor_name = model.make_new_valueinfo_name()
                        intermediate_tensor_shape = model.get_tensor_shape(n.input[0])
                        intermediate_tensor_finn_dtype = model.get_tensor_datatype(
                            mt_cand.output[0]
                        )
                        # Create a new ValueInfoProto and set the shape
                        model.set_tensor_shape(intermediate_tensor_name, intermediate_tensor_shape)
                        # Set the tensor layout
                        model.set_tensor_layout(intermediate_tensor_name, DataLayout.NHWC)
                        # Set the tensor FINN datatype
                        model.set_tensor_datatype(
                            intermediate_tensor_name, intermediate_tensor_finn_dtype
                        )
                        # Rewire output of MT node
                        mt_cand.output[0] = intermediate_tensor_name
                        # Get rid of first transpose node
                        graph.node.remove(n)
                        # Create new Transpose node
                        new_transpose = oh.make_node(
                            "Transpose",
                            [intermediate_tensor_name],
                            [mt_cand_orig_output],
                            perm=[0, 3, 1, 2],
                        )
                        graph.node.insert(node_ind + 1, new_transpose)
                        graph_modified = True
        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class AbsorbTransposeIntoFlatten(Transformation):
    """Absorb transpose node into succeeding flatten node, if H=W=1 and the first
    dimension stays the same. Can also be applied if flatten is implemented implicitly
    by a reshape node with shape [1, -1] and the first input dimension is 1"""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if (
                n.op_type == "Reshape" and (model.get_initializer(n.input[1]) == [1, -1]).all()
            ) or n.op_type == "Flatten":
                prod = model.find_producer(n.input[0])
                if (
                    prod is not None
                    and prod.op_type == "Transpose"
                    # we ensure that the first dimension is not changed from the
                    # transpose operation
                    and get_by_name(prod.attribute, "perm").ints[0] == 0
                ):
                    data_layout = model.get_tensor_layout(prod.input[0])
                    # check for the data layout to interpret input shape correctly
                    if data_layout is None:
                        warnings.warn(
                            """Data layout for input tensor of Transpose node is not set.
                                To use AbsorbTransposeIntoFlatten transformation
                                please set tensor data layout."""
                        )
                        continue
                    elif data_layout == DataLayout.NCHW:
                        (b, c, h, w) = model.get_tensor_shape(prod.input[0])
                        # if h=w=1 the transposition can be absorbed, otherwise
                        # the absorption would lead to an error in the behavior
                        if h != 1 or w != 1:
                            continue
                        # the flatten node from onnx keeps by default the first
                        # dim and flattens the rest, that is why this transformation
                        # can only work with b != 1 if the model contains already a
                        # flatten node and not a reshape node with shape = [1, -1].
                        # If the first  dim of the input tensor is not 1, flatten and
                        # reshape (with shape = [1, -1]) would lead to different results
                        if n.op_type == "Reshape" and b != 1:
                            continue
                    elif data_layout == DataLayout.NHWC:
                        (b, h, w, c) = model.get_tensor_shape(prod.input[0])
                        if h != 1 or w != 1:
                            continue
                        if n.op_type == "Reshape" and b != 1:
                            continue
                    # create single flatten node and remove obsolete nodes
                    node = oh.make_node("Flatten", [prod.input[0]], [n.output[0]])
                    graph.node.remove(n)
                    graph.node.remove(prod)
                    graph.node.insert(node_ind, node)
                    graph_modified = True
        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class AbsorbScalarMulAddIntoTopK(Transformation):
    """Remove mul/add node prior to topk node if the op is scalar. Note that
    the TopK output probabilities will change, but the indices won't."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "TopK":
                prod = model.find_producer(n.input[0])
                if prod is not None and (prod.op_type in ["Mul", "Add"]):
                    prod_input = prod.input[0]
                    param_name = prod.input[1]
                    A = model.get_initializer(param_name)
                    if A is None:
                        warnings.warn("Param is not constant, skipping")
                        continue
                    is_scalar = all(x == 1 for x in A.shape)
                    is_scalar_pos_mul = is_scalar and (prod.op_type == "Mul") and A > 0
                    is_scalar_add = is_scalar and (prod.op_type == "Add")
                    if is_scalar_pos_mul or is_scalar_add:
                        # if the mul is scalar and positive, we can just delete the
                        # mul node and rewire the top k node. Because the top k node
                        # works with probabilities and their relation to each other
                        # the relation doesn't change if every value is multiplied
                        # with a scalar
                        graph.node.remove(prod)
                        n.input[0] = prod_input
                        # to avoid error the dataype is set to float32
                        model.set_tensor_datatype(n.input[0], DataType["FLOAT32"])
                        graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class AbsorbConsecutiveTransposes(Transformation):
    """Remove (Transpose -> Transpose) patterns when the input and output
    of the pattern have the same layout."""

    def are_opposite_permutations(self, perms1, perms2):
        if len(perms1) != len(perms2):
            return False
        assert 0 <= max(perms2) < len(perms2), "invalid permutation"
        assert 0 <= max(perms1) < len(perms1), "invalid permutation"

        for i, p in enumerate(perms2):
            if perms1[p] != i:
                return False

        return True

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for node in graph.node:
            if node.op_type == "Transpose":
                next_nodes = model.find_consumers(node.output[0])
                perms1 = list(get_by_name(node.attribute, "perm").ints)
                if len(next_nodes) == 0:
                    continue
                # check if all nodes after fork are opposite transposes
                all_opposite_transposes = True
                for next_node in next_nodes:
                    if next_node is not None and next_node.op_type == "Transpose":
                        perms2 = list(get_by_name(next_node.attribute, "perm").ints)
                        if not self.are_opposite_permutations(perms1, perms2):
                            all_opposite_transposes = False
                            break
                    else:
                        all_opposite_transposes = False
                        break
                if not all_opposite_transposes:
                    continue
                source_tensor = node.input[0]
                for next_node in next_nodes:
                    # connect next_node's consumers' appropriate input to n's input
                    # TODO how to handle top-level outputs if any?
                    nextnode_out = next_node.output[0]
                    assert nextnode_out not in [x.name for x in model.graph.output]
                    consumers = model.find_consumers(nextnode_out)
                    for cons in consumers:
                        for i, iname in enumerate(cons.input):
                            if iname == nextnode_out:
                                cons.input[i] = source_tensor
                    # remove consumer transpose
                    graph.node.remove(next_node)
                # remove producer transpose
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class AbsorbTransposeIntoResize(Transformation):
    """For (NCHWTranspose -> Resize) move Transpose past Resize and
    change the Resize node's attributes accordingly."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "Transpose" and not model.is_fork_node(node):
                perms = list(get_by_name(node.attribute, "perm").ints)
                if perms == [0, 3, 1, 2]:
                    mt_cand = model.find_consumer(node.output[0])
                    if mt_cand is not None and mt_cand.op_type == "Resize":
                        mode = get_by_name(mt_cand.attribute, "mode").s.decode("ascii")
                        # skip if mode is not nearest
                        if mode != "nearest":
                            continue
                        # if sizes specified, turn into scales
                        if len(mt_cand.input) > 3:
                            sizes = model.get_initializer(mt_cand.input[3])
                        else:
                            sizes = None
                        if sizes is not None:
                            ishape = model.get_tensor_shape(mt_cand.input[0])
                            ns, cs, hs, ws = sizes / np.asarray(ishape)
                            model.set_initializer(mt_cand.input[2], np.asarray([ns, cs, hs, ws]))
                            mt_cand.input.remove(mt_cand.input[3])
                        # scales already specified, transpose indices to NHWC
                        scales = model.get_initializer(mt_cand.input[2])
                        assert scales is not None
                        ns, cs, hs, ws = scales
                        model.set_initializer(mt_cand.input[2], np.asarray([ns, hs, ws, cs]))
                        # get rid of first tranpose node
                        mt_cand.input[0] = node.input[0]
                        graph.node.remove(node)
                        is_last_node = mt_cand.output[0] in [x.name for x in model.graph.output]

                        new_tensor_name = model.make_new_valueinfo_name()
                        if is_last_node:
                            trans_input = new_tensor_name
                            trans_output = mt_cand.output[0]
                        else:
                            trans_input = mt_cand.output[0]
                            trans_output = new_tensor_name
                        # fix tensor shapes for Resize and Transpose
                        n, c, hx, wx = model.get_tensor_shape(mt_cand.output[0])
                        model.set_tensor_shape(trans_input, (n, hx, wx, c))
                        model.set_tensor_shape(trans_output, (n, c, hx, wx))
                        # re-insert Transpose behind Resize
                        new_transpose = oh.make_node(
                            "Transpose",
                            [trans_input],
                            [trans_output],
                            perm=[0, 3, 1, 2],
                        )
                        # rewire nodes
                        final_t_cands = model.find_consumers(mt_cand.output[0])
                        # rewire next nodes' inputs
                        for final_t_cand in final_t_cands:
                            final_t_cand.input[0] = trans_output
                        mt_cand.output[0] = trans_input
                        graph.node.insert(node_ind + 1, new_transpose)
                        graph_modified = True
        if graph_modified:
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class AbsorbMulAddIntoMultiThreshold(Transformation):
    """Absorb preceding Mul->Add chains into MultiThreshold.

    Pattern:
      x --Mul(A)--> y --Add(B)--> z --MultiThreshold(T)-->
    becomes:
      x ---------> MultiThreshold((T - B) / A) -------->

    Supports scalar / 1D / channel-broadcast constants.
    Only positive A is supported.
    """

    def _expand_to_channels(self, arr, num_channels):
        arr = _as_channel_vector(arr)
        if arr is None:
            return None

        arr = np.asarray(arr)

        if arr.ndim == 0:
            return np.full((num_channels,), float(arr), dtype=np.float32)

        if arr.ndim == 1 and len(arr) == num_channels:
            return arr.astype(np.float32)

        return None

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        for mul_node in list(graph.node):
            if mul_node.op_type != "Mul":
                continue

            add_node = _get_single_consumer(model, mul_node.output[0])
            if add_node is None or add_node.op_type != "Add":
                continue

            mt_node = _get_single_consumer(model, add_node.output[0])
            if mt_node is None or mt_node.op_type != "MultiThreshold":
                continue

            mul_const_name, A, mul_start_name = _get_const_and_dyn_input(model, mul_node)
            add_const_name, B, add_start_name = _get_const_and_dyn_input(model, add_node)

            if A is None or B is None:
                continue

            # Add 的动态输入必须来自前一个 Mul
            if add_start_name != mul_node.output[0]:
                continue

            threshold_name = mt_node.input[1]
            T = model.get_initializer(threshold_name)
            if T is None:
                continue

            num_channels = T.shape[0]
            Avec = self._expand_to_channels(A, num_channels)
            Bvec = self._expand_to_channels(B, num_channels)

            if Avec is None or Bvec is None:
                continue

            if np.any(Avec <= 0):
                continue

            Tnew = (T - Bvec.reshape(-1, 1)) / Avec.reshape(-1, 1)
            model.set_initializer(threshold_name, Tnew)

            mt_node.input[0] = mul_start_name

            graph.node.remove(add_node)
            graph.node.remove(mul_node)
            graph_modified = True

        return (model, graph_modified)
    
class DuplicateScalarMulAfterFork(Transformation):
    """Duplicate scalar Mul across all outgoing branches.

    Pattern:
        X -> Mul(s) -> Y1
                  -> Y2
                  -> ...
    becomes:
        X -> Mul_1(s) -> Y1
        X -> Mul_2(s) -> Y2
        ...
    and removes the original shared Mul node.

    Only supports scalar constant multiplier.
    """

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        for n in list(graph.node):
            if n.op_type != "Mul":
                continue

            if len(n.input) != 2:
                continue

            # 找出哪个输入是常量，哪个是动态输入
            const_name, const_val, dyn_name = _get_const_and_dyn_input(model, n)
            if const_val is None:
                continue

            # 只处理 scalar
            arr = np.asarray(const_val)
            is_scalar = (arr.ndim == 0) or all(x == 1 for x in arr.shape)
            if not is_scalar:
                continue

            consumers = model.find_consumers(n.output[0]) or []
            if len(consumers) <= 1:
                continue

            old_out = n.output[0]

            # 给每个 consumer 复制一个新的 Mul
            for cons in consumers:
                # 找到 consumer 里用 old_out 的那个输入位置
                use_idx = None
                for i, inp in enumerate(cons.input):
                    if inp == old_out:
                        use_idx = i
                        break
                if use_idx is None:
                    continue

                new_mul_out = model.make_new_valueinfo_name()

                # 保持 shape / dtype 信息尽量完整
                old_shape = model.get_tensor_shape(old_out)
                if old_shape is not None:
                    model.set_tensor_shape(new_mul_out, old_shape)

                old_dt = model.get_tensor_datatype(old_out)
                if old_dt is not None:
                    model.set_tensor_datatype(new_mul_out, old_dt)

                new_mul = oh.make_node(
                    "Mul",
                    [dyn_name, const_name],
                    [new_mul_out],
                )

                graph.node.append(new_mul)
                cons.input[use_idx] = new_mul_out

            # 删除原来的共享 Mul
            graph.node.remove(n)
            graph_modified = True
            break

        return (model, graph_modified)