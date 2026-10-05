from pathlib import Path
import os
import numpy as np
from onnx import helper as oh

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

from finn.builder.build_dataflow_steps import (
    step_create_dataflow_partition as orig_step_create_dataflow_partition,
)

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.core.datatype import DataType

from qonnx.transformation.base import Transformation
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
    SortGraph,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_data_layouts import InferDataLayouts

from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    AbsorbMulAddIntoMultiThreshold,
    DuplicateScalarMulAfterFork,
    AbsorbConsecutiveTransposes,
)
from finn.transformation.streamline.reorder import (
    FactorOutCommonMulPastAdd,
    MoveTransposePastJoinAdd,
    MoveScalarMulPastPad,
    MoveScalarMulPastIm2Col,
    MoveScalarMulPastMatMul,
    MoveTransposePastFork,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.fpgadataflow import convert_to_hw_layers as to_hw


# =========================================================
# generic helpers
# =========================================================

def _split_const_dyn_input(model, node):
    if len(node.input) != 2:
        return None, None, None
    a_name, b_name = node.input[0], node.input[1]
    a_init = model.get_initializer(a_name)
    b_init = model.get_initializer(b_name)

    if a_init is not None and b_init is None:
        return a_name, a_init, b_name
    elif b_init is not None and a_init is None:
        return b_name, b_init, a_name
    else:
        return None, None, None


def _is_scalar_param(arr):
    arr = np.asarray(arr)
    return (arr.ndim == 0) or all(x == 1 for x in arr.shape)


def _get_perm(node):
    for a in node.attribute:
        if a.name == "perm":
            return list(a.ints)
    return None


def _find_predecessor_by_op(model, node, op_type):
    preds = model.find_direct_predecessors(node)
    if preds is None:
        return None
    for p in preds:
        if p.op_type == op_type:
            return p
    return None


def _find_successor_by_op(model, node, op_type):
    succs = model.find_direct_successors(node)
    if succs is None:
        return None
    for s in succs:
        if s.op_type == op_type:
            return s
    return None


def _find_transpose_predecessor_with_perm(model, node, perm_target):
    preds = model.find_direct_predecessors(node)
    if preds is None:
        return None
    for p in preds:
        if p.op_type != "Transpose":
            continue
        perm = _get_perm(p)
        if perm == perm_target:
            return p
    return None


def _find_transpose_successor_with_perm(model, node, perm_target):
    succs = model.find_direct_successors(node)
    if succs is None:
        return None
    for s in succs:
        if s.op_type != "Transpose":
            continue
        perm = _get_perm(s)
        if perm == perm_target:
            return s
    return None


def _get_const_tensor(model, name):
    try:
        arr = model.get_initializer(name)
        if arr is not None:
            return arr
    except Exception:
        pass

    prod = model.find_producer(name)
    if prod is None or prod.op_type != "Constant":
        return None

    for a in prod.attribute:
        if a.name == "value":
            from onnx import numpy_helper
            return numpy_helper.to_array(a.t)

    return None


def _get_pad_mode(pad_node):
    mode = "constant"
    for a in pad_node.attribute:
        if a.name == "mode":
            try:
                mode = a.s.decode("utf-8")
            except Exception:
                mode = str(a.s)
    return mode


def _get_pad_value(model, pad_node):
    if len(pad_node.input) >= 3:
        cval = _get_const_tensor(model, pad_node.input[2])
        if cval is not None:
            cval = np.asarray(cval).reshape(-1)
            if len(cval) == 0:
                return 0.0
            return float(cval[0])

    for a in pad_node.attribute:
        if a.name == "value":
            return float(a.f)

    return 0.0


def _get_pad_vector(model, pad_node):
    if len(pad_node.input) >= 2:
        pads = _get_const_tensor(model, pad_node.input[1])
        if pads is not None:
            return [int(x) for x in np.asarray(pads).reshape(-1).tolist()]

    for a in pad_node.attribute:
        if a.name == "pads":
            return [int(x) for x in a.ints]

    return None


def _extract_nchw_pad_info(model, pad_node):
    pads = _get_pad_vector(model, pad_node)
    if pads is None or len(pads) != 8:
        return None

    n_beg, c_beg, h_beg, w_beg, n_end, c_end, h_end, w_end = pads

    if n_beg != 0 or c_beg != 0 or n_end != 0 or c_end != 0:
        return None

    return [h_beg, w_beg, h_end, w_end]  # [Top, Left, Bottom, Right]


def _replace_input(node, old_name, new_name):
    for i, x in enumerate(node.input):
        if x == old_name:
            node.input[i] = new_name


# =========================================================
# custom transformations
# =========================================================

class AbsorbScalarMulPadIntoMultiThreshold(Transformation):
    """
    Mul(const scalar) -> Pad(zero) -> MultiThreshold
    =>
    Pad(zero) -> MultiThreshold(threshold / scale)
    """

    @staticmethod
    def _get_attr(node, name, default=None):
        for a in node.attribute:
            if a.name == name:
                return oh.get_attribute_value(a)
        return default

    @staticmethod
    def _get_scalar_initializer(model, tensor_name):
        init = model.get_initializer(tensor_name)
        if init is None:
            return None
        arr = np.asarray(init)
        if arr.size != 1:
            return None
        return float(arr.reshape(-1)[0])

    @staticmethod
    def _get_tensor_consumers(model, tensor_name):
        return [n for n in model.graph.node if tensor_name in n.input]

    @classmethod
    def _pad_is_zero_constant(cls, model, pad_node):
        mode = cls._get_attr(pad_node, "mode", b"constant")
        if isinstance(mode, bytes):
            mode = mode.decode("utf-8")
        if mode != "constant":
            return False

        if len(pad_node.input) >= 3 and pad_node.input[2] != "":
            pad_val = model.get_initializer(pad_node.input[2])
            if pad_val is None:
                return False
            return np.allclose(np.asarray(pad_val), 0.0)

        attr_val = cls._get_attr(pad_node, "value", None)
        if attr_val is not None:
            try:
                return np.allclose(np.asarray(attr_val), 0.0)
            except Exception:
                return False

        return True

    def apply(self, model):
        modified = False
        nodes = list(model.graph.node)

        for mul in nodes:
            if mul.op_type != "Mul":
                continue

            data_inp = None
            const_inp = None
            scale = None

            for inp in mul.input:
                s = self._get_scalar_initializer(model, inp)
                if s is not None:
                    const_inp = inp
                    scale = s
                else:
                    data_inp = inp

            if data_inp is None or const_inp is None or scale is None:
                continue
            if scale <= 0:
                continue

            mul_out = mul.output[0]
            mul_cons = self._get_tensor_consumers(model, mul_out)
            if len(mul_cons) != 1:
                continue

            pad = mul_cons[0]
            if pad.op_type != "Pad":
                continue
            if not self._pad_is_zero_constant(model, pad):
                continue

            pad_out = pad.output[0]
            pad_cons = self._get_tensor_consumers(model, pad_out)
            if len(pad_cons) != 1:
                continue

            mt = pad_cons[0]
            if mt.op_type != "MultiThreshold":
                continue
            if len(mt.input) < 2:
                continue

            th_name = mt.input[1]
            th = model.get_initializer(th_name)
            if th is None:
                continue

            new_th = np.asarray(th).astype(np.float32) / float(scale)
            model.set_initializer(th_name, new_th.astype(np.float32))

            for i, inp in enumerate(pad.input):
                if inp == mul_out:
                    pad.input[i] = data_inp

            if mul in model.graph.node:
                model.graph.node.remove(mul)

            modified = True
            print(
                f"[AbsorbScalarMulPadIntoMultiThreshold] absorbed {mul.name} "
                f"(scale={scale}) into {mt.name}"
            )

        return (model, modified)


class AbsorbScalarMulIntoThresholdPath(Transformation):
    """
    Mul(s) -> [linear ops]* -> MatMul -> MultiThreshold(T)
    =>
    [linear ops]* -> MatMul -> MultiThreshold(T / s)
    """

    def __init__(self, run_round_and_clip=True, debug=False):
        super().__init__()
        self.run_round_and_clip = run_round_and_clip
        self.debug = debug

    def _dbg(self, *args):
        if self.debug:
            print("[AbsorbScalarMulIntoThresholdPath]", *args)

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        allowed_linear_ops = {
            "Pad",
            "Transpose",
            "Reshape",
            "Flatten",
            "Squeeze",
            "Unsqueeze",
            "Im2Col",
        }

        for mul in list(graph.node):
            if mul.op_type != "Mul":
                continue

            const_name, const_val, dyn_name = _split_const_dyn_input(model, mul)
            if const_val is None or not _is_scalar_param(const_val):
                continue

            scale = float(np.asarray(const_val).reshape(-1)[0])
            if scale <= 0.0:
                continue

            cur_tensor = mul.output[0]
            consumer = model.find_consumer(cur_tensor)
            if consumer is None:
                continue

            first_linear = None

            while consumer is not None and consumer.op_type in allowed_linear_ops:
                if model.is_join_node(consumer):
                    break
                if first_linear is None:
                    first_linear = consumer
                cur_tensor = consumer.output[0]
                consumer = model.find_consumer(cur_tensor)

            if consumer is None or consumer.op_type != "MatMul":
                continue
            matmul = consumer

            if model.is_join_node(matmul):
                continue
            if matmul.input[0] != cur_tensor:
                continue

            mt = model.find_consumer(matmul.output[0])
            if mt is None or mt.op_type != "MultiThreshold":
                continue

            if model.is_fork_node(matmul):
                continue

            th_name = mt.input[1]
            T = model.get_initializer(th_name)
            if T is None:
                continue

            model.set_initializer(th_name, T / scale)

            if first_linear is not None:
                first_linear.input[0] = dyn_name
            else:
                matmul.input[0] = dyn_name

            graph.node.remove(mul)
            graph_modified = True
            self._dbg(
                f"absorbed scalar scale={scale} from {mul.name or '<unnamed>'} "
                f"into thresholds of {mt.name or '<unnamed>'}"
            )

        if graph_modified:
            model = model.transform(SortGraph())
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

            if self.run_round_and_clip:
                model = model.transform(RoundAndClipThresholds())
                model = model.transform(InferShapes())
                model = model.transform(InferDataTypes())

        return (model, graph_modified)


class ForceIntegerAnnotationsOnQuantPaths(Transformation):
    """
    Force integer datatype annotations on paths that are mathematically integer again.
    """

    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug

    def _dbg(self, *args):
        if self.debug:
            print("[ForceIntegerAnnotationsOnQuantPaths]", *args)

    def apply(self, model):
        graph_modified = False
        changed = True

        passthrough_ops = {
            "Pad",
            "Transpose",
            "Reshape",
            "Flatten",
            "Squeeze",
            "Unsqueeze",
            "Im2Col",
        }

        while changed:
            changed = False

            for n in model.graph.node:
                if n.op_type in passthrough_ops and len(n.input) >= 1 and len(n.output) >= 1:
                    in_name = n.input[0]
                    out_name = n.output[0]
                    idt = model.get_tensor_datatype(in_name)
                    odt = model.get_tensor_datatype(out_name)

                    if idt is not None and idt.is_integer():
                        if odt is None or (not odt.is_integer()) or (odt != idt):
                            model.set_tensor_datatype(out_name, idt)
                            changed = True
                            graph_modified = True

                elif n.op_type == "MatMul" and len(n.input) == 2 and len(n.output) == 1:
                    a_name = n.input[0]
                    w_name = n.input[1]
                    y_name = n.output[0]

                    adt = model.get_tensor_datatype(a_name)
                    wdt = model.get_tensor_datatype(w_name)
                    ydt = model.get_tensor_datatype(y_name)

                    if (
                        adt is not None
                        and wdt is not None
                        and adt.is_integer()
                        and wdt.is_integer()
                    ):
                        if ydt is None or (not ydt.is_integer()) or (ydt != DataType["INT32"]):
                            model.set_tensor_datatype(y_name, DataType["INT32"])
                            changed = True
                            graph_modified = True

                elif n.op_type == "MultiThreshold" and len(n.input) >= 2:
                    th_name = n.input[1]
                    tdt = model.get_tensor_datatype(th_name)
                    T = model.get_initializer(th_name)

                    if T is not None and np.allclose(T, np.round(T)):
                        if tdt is None or (not tdt.is_integer()):
                            model.set_tensor_datatype(th_name, DataType["INT32"])
                            changed = True
                            graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

        return (model, graph_modified)


class LowerPadSandwichToFMPadding(Transformation):
    """
    Transpose [0,3,1,2] -> Pad(NCHW, zero) -> Transpose [0,2,3,1]
    =>
    FMPadding in NHWC
    """

    def apply(self, model: ModelWrapper):
        graph = model.graph
        modified = False

        for pad in list(graph.node):
            if pad.op_type != "Pad":
                continue

            if _get_pad_mode(pad) != "constant":
                continue
            if _get_pad_value(model, pad) != 0.0:
                continue

            t1 = _find_transpose_predecessor_with_perm(model, pad, [0, 3, 1, 2])
            if t1 is None:
                continue

            t2 = _find_transpose_successor_with_perm(model, pad, [0, 2, 3, 1])
            if t2 is None:
                continue

            pad_tlbr = _extract_nchw_pad_info(model, pad)
            if pad_tlbr is None:
                continue

            inp_name = t1.input[0]
            out_name = t2.output[0]

            ishape = model.get_tensor_shape(inp_name)
            if ishape is None or len(ishape) != 4:
                continue

            n, h, w, c = [int(x) for x in ishape]
            pad_t, pad_l, pad_b, pad_r = pad_tlbr
            oshape = [n, h + pad_t + pad_b, w + pad_l + pad_r, c]

            in_dt = model.get_tensor_datatype(inp_name)
            in_dt_name = in_dt.name if hasattr(in_dt, "name") else str(in_dt)

            print(
                "Matched pad sandwich:",
                t1.name if t1.name else "<unnamed>",
                "->",
                pad.name if pad.name else "<unnamed>",
                "->",
                t2.name if t2.name else "<unnamed>",
                "| pad tlbr =",
                [pad_t, pad_l, pad_b, pad_r],
                "| input shape =",
                ishape,
            )

            new_node = oh.make_node(
                "FMPadding",
                inputs=[inp_name],
                outputs=[out_name],
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                ImgDim=[int(h), int(w)],
                Padding=[int(pad_t), int(pad_l), int(pad_b), int(pad_r)],
                NumChannels=int(c),
                SIMD=int(c),
                inputDataType=in_dt_name,
                name="FMPadding_" + (pad.name if pad.name else "auto"),
            )

            insert_idx = list(graph.node).index(t1)
            graph.node.insert(insert_idx, new_node)

            graph.node.remove(t1)
            graph.node.remove(pad)
            graph.node.remove(t2)

            model.set_tensor_shape(out_name, oshape)
            model.set_tensor_datatype(out_name, in_dt)

            modified = True

        return (model, modified)


class LowerInputForkPadToFMPadding(Transformation):
    """
    DuplicateStreams
      ├─ Transpose([0,2,3,1]) -> MVAU
      └─ Pad(NCHW, const 0) -> Transpose([0,2,3,1]) -> ConvolutionInputGenerator
    =>
    Transpose_pre([0,2,3,1]) -> DuplicateStreams
      ├─ -> MVAU
      └─ -> FMPadding -> ConvolutionInputGenerator
    """

    def apply(self, model: ModelWrapper):
        graph = model.graph
        modified = False

        for pad in list(graph.node):
            if pad.op_type != "Pad":
                continue

            if _get_pad_mode(pad) != "constant":
                continue
            if _get_pad_value(model, pad) != 0.0:
                continue

            t_pad = _find_successor_by_op(model, pad, "Transpose")
            if t_pad is None or _get_perm(t_pad) != [0, 2, 3, 1]:
                continue

            dup = _find_predecessor_by_op(model, pad, "DuplicateStreams")
            if dup is None:
                continue

            t_direct = None
            succs = model.find_direct_successors(dup)
            if succs is None:
                continue

            for s in succs:
                if s == pad:
                    continue
                if s.op_type == "Transpose" and _get_perm(s) == [0, 2, 3, 1]:
                    t_direct = s
                    break

            if t_direct is None:
                continue

            mvau0 = _find_successor_by_op(model, t_direct, "MVAU")
            cig0 = _find_successor_by_op(model, t_pad, "ConvolutionInputGenerator")
            if mvau0 is None or cig0 is None:
                continue

            old_dup_in = dup.input[0]
            ishape_nchw = model.get_tensor_shape(old_dup_in)
            if ishape_nchw is None or len(ishape_nchw) != 4:
                continue

            n, c, h, w = [int(x) for x in ishape_nchw]
            ishape_nhwc = [n, h, w, c]

            pad_tlbr = _extract_nchw_pad_info(model, pad)
            if pad_tlbr is None:
                continue
            pad_t, pad_l, pad_b, pad_r = pad_tlbr

            dup_dt = model.get_tensor_datatype(old_dup_in)
            dup_dt_name = dup_dt.name if hasattr(dup_dt, "name") else str(dup_dt)

            print(
                "Matched input fork pattern:",
                dup.name if dup.name else "<unnamed>",
                "| direct branch =", t_direct.name if t_direct.name else "<unnamed>",
                "| pad branch =", pad.name if pad.name else "<unnamed>",
                "->",
                t_pad.name if t_pad.name else "<unnamed>",
                "| old dup in =", ishape_nchw,
                "| new dup in =", ishape_nhwc,
            )

            nhwc_in_name = old_dup_in + "_nhwc"
            t_pre = oh.make_node(
                "Transpose",
                inputs=[old_dup_in],
                outputs=[nhwc_in_name],
                perm=[0, 2, 3, 1],
                name="Transpose_pre_" + (dup.name if dup.name else "dup"),
            )

            dup_insert_idx = list(graph.node).index(dup)
            graph.node.insert(dup_insert_idx, t_pre)

            model.set_tensor_shape(nhwc_in_name, ishape_nhwc)
            model.set_tensor_datatype(nhwc_in_name, dup_dt)

            dup.input[0] = nhwc_in_name

            dup_inst = getCustomOp(dup)
            dup_inst.set_nodeattr("NumChannels", int(c))
            dup_inst.set_nodeattr("numInputVectors", [int(n), int(h), int(w)])
            dup_inst.set_nodeattr("PE", 1)
            try:
                dup_inst.set_nodeattr("inputDataType", dup_dt_name)
            except Exception:
                pass

            for out_name in dup.output:
                model.set_tensor_shape(out_name, ishape_nhwc)
                model.set_tensor_datatype(out_name, dup_dt)

            _replace_input(mvau0, t_direct.output[0], t_direct.input[0])

            fmpad_out = t_pad.output[0]
            fmpad_shape = [n, h + pad_t + pad_b, w + pad_l + pad_r, c]

            fmpad = oh.make_node(
                "FMPadding",
                inputs=[pad.input[0]],
                outputs=[fmpad_out],
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                ImgDim=[int(h), int(w)],
                Padding=[int(pad_t), int(pad_l), int(pad_b), int(pad_r)],
                NumChannels=int(c),
                SIMD=int(c),
                inputDataType=dup_dt_name,
                name="FMPadding_input_" + (pad.name if pad.name else "auto"),
            )

            pad_insert_idx = list(graph.node).index(pad)
            graph.node.insert(pad_insert_idx, fmpad)

            model.set_tensor_shape(fmpad_out, fmpad_shape)
            model.set_tensor_datatype(fmpad_out, dup_dt)

            graph.node.remove(t_direct)
            graph.node.remove(pad)
            graph.node.remove(t_pad)

            modified = True

        return (model, modified)


# =========================================================
# custom build steps
# =========================================================

def step_tidy_up_no_fold(model, cfg):
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    return model


def step_round_thresholds(model: ModelWrapper, cfg):
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    return model


def step_fix_names_after_specialize(model, cfg):
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


def step_fix_pad_integer_dtypes(model, cfg):
    changed = 0

    for node in model.graph.node:
        if node.op_type != "Pad":
            continue

        inp = node.input[0]
        out = node.output[0]

        try:
            in_dt = model.get_tensor_datatype(inp)
        except Exception:
            in_dt = None

        if in_dt is None:
            continue

        try:
            if in_dt.is_integer():
                model.set_tensor_datatype(out, in_dt)
                changed += 1
                print(f"[step_fix_pad_integer_dtypes] {out} <- {in_dt.name}")
        except Exception as e:
            print(f"[step_fix_pad_integer_dtypes] skip {out}: {e}")

    print(f"[step_fix_pad_integer_dtypes] changed={changed}")
    return model


def step_absorb_mul_pad_mt_patched(model, cfg):
    print("[PATCH] absorbing Mul -> Pad -> MultiThreshold float islands")

    prev_nodes = -1
    it = 0

    while prev_nodes != len(model.graph.node) and it < 10:
        prev_nodes = len(model.graph.node)
        it += 1

        model = model.transform(AbsorbScalarMulPadIntoMultiThreshold())

        try:
            model = model.transform(RoundAndClipThresholds())
        except Exception as e:
            print("[PATCH] RoundAndClipThresholds warning:", e)

        try:
            model = model.transform(AbsorbConsecutiveTransposes())
        except Exception:
            pass

        try:
            model = model.transform(InferShapes())
        except Exception as e:
            print("[PATCH] InferShapes warning:", e)

        try:
            model = model.transform(InferDataTypes())
        except Exception as e:
            print("[PATCH] InferDataTypes warning:", e)

    dbg_dir = cfg.output_dir + "/intermediate_models"
    os.makedirs(dbg_dir, exist_ok=True)
    dbg_path = dbg_dir + "/step_absorb_mul_pad_mt_patched.onnx"
    model.save(dbg_path)
    print("[PATCH] saved absorb-mul-pad-mt checkpoint to:", dbg_path)

    return model


def step_after_streamline_cleanup(model, cfg):
    print("[PATCH] cleanup between step_streamline and step_convert_to_hw")

    prev_nodes = -1
    it = 0
    max_iter = 20

    while prev_nodes != len(model.graph.node) and it < max_iter:
        prev_nodes = len(model.graph.node)
        it += 1

        model = model.transform(AbsorbConsecutiveTransposes())
        model = model.transform(FactorOutCommonMulPastAdd())
        model = model.transform(
            AbsorbScalarMulIntoThresholdPath(
                run_round_and_clip=True,
                debug=True,
            )
        )
        model = model.transform(AbsorbMulIntoMultiThreshold())
        model = model.transform(AbsorbAddIntoMultiThreshold())
        model = model.transform(AbsorbSignBiasIntoMultiThreshold())
        model = model.transform(ForceIntegerAnnotationsOnQuantPaths(debug=True))
        model = model.transform(AbsorbConsecutiveTransposes())
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

    dbg_dir = cfg.output_dir + "/intermediate_models"
    os.makedirs(dbg_dir, exist_ok=True)
    model.save(dbg_dir + "/step_after_streamline_cleanup.onnx")
    return model


def step_convert_to_hw_patched(model, cfg):
    if cfg.standalone_thresholds:
        model = model.transform(to_hw.InferThresholdingLayer())

    model = model.transform(to_hw.InferBinaryMatrixVectorActivation())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(to_hw.InferThresholdingLayer())

    if len(model.get_nodes_by_op_type("Im2Col")) > 0:
        model = model.transform(to_hw.InferConvInpGen())

    model = model.transform(to_hw.InferStreamingMaxPool())
    model = model.transform(AbsorbConsecutiveTransposes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferDataLayouts())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


def step_after_convert_to_hw_cleanup(model, cfg):
    print("[PATCH] cleanup immediately after step_convert_to_hw")

    prev_nodes = -1
    it = 0
    max_iter = 10

    while prev_nodes != len(model.graph.node) and it < max_iter:
        prev_nodes = len(model.graph.node)
        it += 1

        try:
            model = model.transform(MoveTransposePastFork())
        except Exception as e:
            print("[WARN] MoveTransposePastFork:", e)

        model = model.transform(AbsorbConsecutiveTransposes())
        model = model.transform(to_hw.InferDuplicateStreamsLayer())
        model = model.transform(to_hw.InferAddStreamsLayer())

        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        model = model.transform(InferDataLayouts())

    print("[PATCH] lowering pad/transpose patterns to FMPadding")

    prev_nodes = -1
    it = 0
    max_iter = 8

    while prev_nodes != len(model.graph.node) and it < max_iter:
        prev_nodes = len(model.graph.node)
        it += 1

        model = model.transform(LowerPadSandwichToFMPadding())
        model = model.transform(LowerInputForkPadToFMPadding())

        model = model.transform(AbsorbConsecutiveTransposes())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        model = model.transform(InferDataLayouts())

    dbg_dir = cfg.output_dir + "/intermediate_models"
    os.makedirs(dbg_dir, exist_ok=True)
    dbg_path = dbg_dir + "/step_after_convert_to_hw_cleanup.onnx"
    model.save(dbg_path)
    print("[PATCH] saved post-convert cleanup checkpoint to:", dbg_path)

    return model


def step_create_dataflow_partition_patched(model, cfg):
    print("[PATCH] final normalization before dataflow partition")

    model = model.transform(AbsorbConsecutiveTransposes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())

    dbg_dir = cfg.output_dir + "/intermediate_models"
    os.makedirs(dbg_dir, exist_ok=True)
    dbg_path = dbg_dir + "/pre_partition_cleanup.onnx"
    model.save(dbg_path)
    print("[PATCH] saved cleanup checkpoint to:", dbg_path)

    return orig_step_create_dataflow_partition(model, cfg)


# =========================================================
# build config
# =========================================================

BASE_DIR = Path("/home/slowman/Desktop/project/Thesis/finn/notebooks/pytorch-tcn")
MODEL_FILE = BASE_DIR / "onnx_exports" / "ecg5000_static_qtcn_finn_streamlined.onnx"
OUT_DIR = BASE_DIR / "finn_build_pynqz2"

custom_steps = [
    "step_streamline",
    step_after_streamline_cleanup,
    step_absorb_mul_pad_mt_patched,
    step_round_thresholds,
    step_fix_pad_integer_dtypes,
    step_convert_to_hw_patched,
    step_after_convert_to_hw_cleanup,
    step_create_dataflow_partition_patched,
    "step_specialize_layers",
    step_fix_names_after_specialize,
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    #"step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    "step_synthesize_bitfile",
    "step_make_pynq_driver",
    "step_deployment_package",
]

cfg = build_cfg.DataflowBuildConfig(
    output_dir=str(OUT_DIR),
    board="Pynq-Z2",
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    synth_clk_period_ns=10.0,
    steps=custom_steps,
    save_intermediate_models=True,
    verbose=True,
    specialize_layers_config_file=str(
        BASE_DIR / "finn_build_pynqz2" / "template_specialize_layers_config_customized.json"
    ),
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ],
)

build.build_dataflow_cfg(str(MODEL_FILE), cfg)