from pathlib import Path
import os

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

from finn.builder.build_dataflow_steps import (
    step_create_dataflow_partition as orig_step_create_dataflow_partition,
    step_convert_to_hw as orig_step_convert_to_hw,
)

from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    AbsorbMulAddIntoMultiThreshold,
    DuplicateScalarMulAfterFork,
    AbsorbConsecutiveTransposes,
)

from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

import numpy as np
from onnx import helper as oh
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
#from qonnx.transformation.infer_shapes import InferShapes
#from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.core.modelwrapper import ModelWrapper

from finn.transformation.streamline.reorder import FactorOutCommonMulPastAdd
from finn.transformation.streamline.reorder import MoveTransposePastJoinAdd,MoveScalarMulPastPad,MoveScalarMulPastIm2Col,MoveScalarMulPastMatMul
#from finn.transformation.streamline.absorb import AbsorbConsecutiveTransposes

from qonnx.core.datatype import DataType
from qonnx.transformation.infer_data_layouts import InferDataLayouts

from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.fpgadataflow import convert_to_hw_layers as to_hw
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferAddStreamsLayer,
    InferDuplicateStreamsLayer,
    InferChannelwiseLinearLayer,
    InferBinaryMatrixVectorActivation,
    InferQuantizedMatrixVectorActivation,
    InferThresholdingLayer,
)

from finn.transformation.streamline.reorder import MoveTransposePastFork

# def _split_const_dyn_input(model, node):
#     if len(node.input) != 2:
#         return None, None, None
#     a_name, b_name = node.input[0], node.input[1]
#     a_init = model.get_initializer(a_name)
#     b_init = model.get_initializer(b_name)

#     if a_init is not None and b_init is None:
#         return a_name, a_init, b_name
#     elif b_init is not None and a_init is None:
#         return b_name, b_init, a_name
#     else:
#         return None, None, None


# def _is_scalar_param(arr):
#     arr = np.asarray(arr)
#     return (arr.ndim == 0) or all(x == 1 for x in arr.shape)
def step_round_thresholds(model: ModelWrapper, cfg):
    """Round MultiThreshold thresholds to integer after streamlining."""
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    return model

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


def step_create_dataflow_partition_patched(model, cfg):
    print("[PATCH] re-running full step_convert_to_hw after cleanup")
    model = orig_step_convert_to_hw(model, cfg)

    print("[PATCH] re-running reorder + HW inference passes after full convert_to_hw")

    post_prev = -1
    post_it = 0
    it = 0
    while post_prev != len(model.graph.node) and it < 10:
        post_prev = len(model.graph.node)
        it += 1

        try:
            model = model.transform(MoveTransposePastJoinAdd())
        except Exception:
            pass

        model = model.transform(AbsorbConsecutiveTransposes())

    # residual add 前面的共同 scale 提到后面
        model = model.transform(FactorOutCommonMulPastAdd())

    # 把卷积分支上的 scalar mul 往后推
        model = model.transform(MoveScalarMulPastPad())
        model = model.transform(MoveScalarMulPastIm2Col())
        model = model.transform(MoveScalarMulPastMatMul())

    # 再吸收到 MultiThreshold
        model = model.transform(AbsorbMulIntoMultiThreshold())
        model = model.transform(AbsorbAddIntoMultiThreshold())
        model = model.transform(AbsorbSignBiasIntoMultiThreshold())

    # 新长出来的 thresholds 重新 round/clip 成整数
        model = model.transform(RoundAndClipThresholds())

        model = model.transform(AbsorbConsecutiveTransposes())
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

    print("[PATCH] re-running full step_convert_to_hw after cleanup")
    model = orig_step_convert_to_hw(model, cfg)

    print("[PATCH] re-running selected HW inference passes after full convert_to_hw")

    post_prev = -1
    post_it = 0
    while post_prev != len(model.graph.node) and post_it < 10:
        post_prev = len(model.graph.node)
        post_it += 1

        try:
            model = model.transform(MoveLinearPastEltwiseAdd())
        except Exception:
            pass

        try:
            model = model.transform(MoveLinearPastFork())
        except Exception:
            pass

        model = model.transform(FactorOutCommonMulPastAdd())
        model = model.transform(MoveScalarMulPastPad())
        model = model.transform(MoveScalarMulPastIm2Col())
        model = model.transform(MoveScalarMulPastMatMul())
        model = model.transform(AbsorbMulIntoMultiThreshold())
        model = model.transform(RoundAndClipThresholds())

        model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
        model = model.transform(to_hw.InferThresholdingLayer())
        model = model.transform(to_hw.InferAddStreamsLayer())
        model = model.transform(to_hw.InferDuplicateStreamsLayer())
        model = model.transform(to_hw.InferChannelwiseLinearLayer())

        model = model.transform(AbsorbConsecutiveTransposes())
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

    dbg_dir = cfg.output_dir + "/intermediate_models"
    os.makedirs(dbg_dir, exist_ok=True)
    dbg_path = dbg_dir + "/pre_partition_cleanup.onnx"
    model.save(dbg_path)
    print("[PATCH] saved cleanup checkpoint to:", dbg_path)

    #return orig_step_create_dataflow_partition(model, cfg)
    raise RuntimeError("DEBUG STOP AFTER pre_partition_cleanup")


def step_tidy_up_no_fold(model, cfg):
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    return model


def step_fix_names_after_specialize(model, cfg):
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model

class AbsorbScalarMulIntoThresholdPath(Transformation):
    """
    Generalized version.

    Absorb:
        Mul(s) -> [linear ops]* -> MatMul -> MultiThreshold(T)

    into:
        [linear ops]* -> MatMul -> MultiThreshold(T / s)

    where [linear ops] can include:
        Pad, Transpose, Reshape, Flatten, Squeeze, Unsqueeze, Im2Col

    Assumptions:
    - scalar positive scale s
    - linear single-consumer chain
    - MatMul consumes the chain on input 0
    - MultiThreshold immediately follows MatMul
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

            # must be a simple scalar-mul node
            const_name, const_val, dyn_name = _split_const_dyn_input(model, mul)
            if const_val is None or not _is_scalar_param(const_val):
                continue

            scale = float(np.asarray(const_val).reshape(-1)[0])
            if scale <= 0.0:
                continue

            # Start from Mul output, walk through a single-consumer linear chain
            cur_tensor = mul.output[0]
            consumer = model.find_consumer(cur_tensor)
            if consumer is None:
                continue

            first_linear = None
            last_linear = None

            while consumer is not None and consumer.op_type in allowed_linear_ops:
                if model.is_join_node(consumer):
                    break
                if first_linear is None:
                    first_linear = consumer
                last_linear = consumer
                cur_tensor = consumer.output[0]
                consumer = model.find_consumer(cur_tensor)

            # after the linear chain, we expect MatMul
            if consumer is None or consumer.op_type != "MatMul":
                continue
            matmul = consumer

            if model.is_join_node(matmul):
                continue

            # MatMul must consume the walked path on input 0
            if matmul.input[0] != cur_tensor:
                continue

            mt = model.find_consumer(matmul.output[0])
            if mt is None or mt.op_type != "MultiThreshold":
                continue

            if model.is_fork_node(matmul):
                continue

            # get thresholds
            th_name = mt.input[1]
            T = model.get_initializer(th_name)
            if T is None:
                continue

            # absorb scale into thresholds
            Tnew = T / scale
            model.set_initializer(th_name, Tnew)

            # bypass Mul:
            # if there is a linear op chain, rewire first one;
            # else MatMul consumes dyn input directly
            if first_linear is not None:
                old_in = first_linear.input[0]
                first_linear.input[0] = dyn_name
                self._dbg(
                    f"{mul.name or '<unnamed>'}: rewired "
                    f"{first_linear.name or first_linear.op_type}.input[0] "
                    f"from {old_in} to {dyn_name}"
                )
            else:
                old_in = matmul.input[0]
                matmul.input[0] = dyn_name
                self._dbg(
                    f"{mul.name or '<unnamed>'}: rewired "
                    f"{matmul.name or 'MatMul'}.input[0] "
                    f"from {old_in} to {dyn_name}"
                )

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
    After scalar-mul absorption, many tensors are mathematically integer-valued
    again, but their datatype annotations may still remain FLOAT32.

    This pass force-propagates integer datatypes through linear data-movement ops
    and fixes MatMul/MultiThreshold annotations so FINN HW inference can match.

    What it does:
      - Pad / Transpose / Reshape / Flatten / Squeeze / Unsqueeze / Im2Col:
          if input dtype is integer, force output dtype = input dtype
      - MatMul:
          if input dtype and weight dtype are integer, force output dtype = INT32
      - MultiThreshold threshold tensor:
          if initializer values are integer-valued but dtype annotation is float,
          force threshold tensor dtype = INT32
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
                # -------------------------------------------------
                # 1) propagate integer dtype through linear ops
                # -------------------------------------------------
                if n.op_type in passthrough_ops and len(n.input) >= 1 and len(n.output) >= 1:
                    in_name = n.input[0]
                    out_name = n.output[0]
                    idt = model.get_tensor_datatype(in_name)
                    odt = model.get_tensor_datatype(out_name)

                    if idt is not None and idt.is_integer():
                        if odt is None or (not odt.is_integer()) or (odt != idt):
                            model.set_tensor_datatype(out_name, idt)
                            self._dbg(
                                f"{n.name or n.op_type}: set {out_name} dtype "
                                f"from {odt} to {idt}"
                            )
                            changed = True
                            graph_modified = True

                # -------------------------------------------------
                # 2) force MatMul output dtype to INT32 when inputs are integer
                # -------------------------------------------------
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
                            self._dbg(
                                f"{n.name or 'MatMul'}: set {y_name} dtype "
                                f"from {ydt} to INT32"
                            )
                            changed = True
                            graph_modified = True

                # -------------------------------------------------
                # 3) if MultiThreshold thresholds are integer-valued numerically,
                #    fix their dtype annotation to INT32
                # -------------------------------------------------
                elif n.op_type == "MultiThreshold" and len(n.input) >= 2:
                    th_name = n.input[1]
                    tdt = model.get_tensor_datatype(th_name)
                    T = model.get_initializer(th_name)

                    if T is not None:
                        # after RoundAndClipThresholds, values are typically integers
                        if np.allclose(T, np.round(T)):
                            if tdt is None or (not tdt.is_integer()):
                                model.set_tensor_datatype(th_name, DataType["INT32"])
                                self._dbg(
                                    f"{n.name or 'MultiThreshold'}: set threshold {th_name} "
                                    f"dtype from {tdt} to INT32"
                                )
                                changed = True
                                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

        return (model, graph_modified)
        
class MoveScalarMulPastConvPipeline(Transformation):
    """x -> Mul(a) -> Pad -> Im2Col -> MatMul -> y
       becomes
       x -> Pad -> Im2Col -> MatMul -> Mul(a) -> y
    """

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        for n in list(graph.node):
            if n.op_type != "Mul":
                continue
            if model.is_fork_node(n) or model.is_join_node(n):
                continue

            const_name, const_val, dyn_name = _split_const_dyn_input(model, n)
            if const_val is None or not _is_scalar_param(const_val):
                continue

            pad = model.find_consumer(n.output[0])
            if pad is None or pad.op_type != "Pad" or model.is_join_node(pad):
                continue

            im2col = model.find_consumer(pad.output[0])
            if im2col is None or im2col.op_type != "Im2Col" or model.is_join_node(im2col):
                continue

            matmul = model.find_consumer(im2col.output[0])
            if matmul is None or matmul.op_type != "MatMul" or model.is_join_node(matmul):
                continue

            if matmul.input[0] != im2col.output[0]:
                continue

            old_mm_out = matmul.output[0]
            mm_out_shape = model.get_tensor_shape(old_mm_out)
            mm_out_layout = model.get_tensor_layout(old_mm_out)

            mid_name = model.make_new_valueinfo_name()
            if mm_out_shape is not None:
                model.set_tensor_shape(mid_name, mm_out_shape)
            if mm_out_layout is not None:
                model.set_tensor_layout(mid_name, mm_out_layout)

            # Mul 从最前面拿掉
            pad.input[0] = dyn_name

            # MatMul 输出改成中间张量
            matmul.output[0] = mid_name

            # 在 MatMul 后面重新插入 Mul
            insert_idx = list(graph.node).index(matmul)
            new_mul = oh.make_node(
                "Mul",
                [mid_name, const_name],
                [old_mm_out],
                name=n.name + "_moved",
            )
            graph.node.insert(insert_idx + 1, new_mul)

            # 删除旧的 Mul
            graph.node.remove(n)
            graph_modified = True

        if graph_modified:
            model = model.transform(SortGraph())
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

from finn.transformation.streamline.absorb import (
    AbsorbMulIntoMultiThreshold,
    AbsorbAddIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    AbsorbConsecutiveTransposes,
)
from finn.transformation.streamline.reorder import (
    FactorOutCommonMulPastAdd,
    MoveScalarMulPastMatMul,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds


def step_after_streamline_cleanup(model, cfg):
    print("[PATCH] cleanup between step_streamline and step_convert_to_hw")

    prev_nodes = -1
    it = 0
    max_iter = 20

    while prev_nodes != len(model.graph.node) and it < max_iter:
        prev_nodes = len(model.graph.node)
        it += 1

        model = model.transform(AbsorbConsecutiveTransposes())

        # residual branches
        model = model.transform(FactorOutCommonMulPastAdd())

        # absorb pre-conv scalar mul into downstream thresholds
        model = model.transform(
            AbsorbScalarMulIntoThresholdPath(
                run_round_and_clip=True,
                debug=True,
            )
        )

        # residual / add 周围继续收
        model = model.transform(AbsorbMulIntoMultiThreshold())
        model = model.transform(AbsorbAddIntoMultiThreshold())
        model = model.transform(AbsorbSignBiasIntoMultiThreshold())

        # 关键：修正整数链上的 datatype annotation
        model = model.transform(
            ForceIntegerAnnotationsOnQuantPaths(debug=True)
        )

        model = model.transform(AbsorbConsecutiveTransposes())
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

    dbg_dir = cfg.output_dir + "/intermediate_models"
    os.makedirs(dbg_dir, exist_ok=True)
    model.save(dbg_dir + "/step_after_streamline_cleanup.onnx")
    return model

# def step_after_convert_to_hw_cleanup(model, cfg):
#     print("[PATCH] cleanup immediately after step_convert_to_hw")

#     prev_nodes = -1
#     it = 0
#     max_iter = 8

#     while prev_nodes != len(model.graph.node) and it < max_iter:
#         prev_nodes = len(model.graph.node)
#         it += 1

#         # 只把 residual add / fork 变成 HW nodes
#         model = model.transform(to_hw.InferAddStreamsLayer())
#         model = model.transform(to_hw.InferDuplicateStreamsLayer())

#         # 收掉一些多余 transpose
#         model = model.transform(AbsorbConsecutiveTransposes())

#         model = model.transform(InferShapes())
#         model = model.transform(InferDataTypes())

#     dbg_dir = cfg.output_dir + "/intermediate_models"
#     os.makedirs(dbg_dir, exist_ok=True)
#     dbg_path = dbg_dir + "/step_after_convert_to_hw_cleanup.onnx"
#     model.save(dbg_path)
#     print("[PATCH] saved post-convert cleanup checkpoint to:", dbg_path)

#     return model
    
def step_after_convert_to_hw_cleanup(model, cfg):
    print("[PATCH] cleanup immediately after step_convert_to_hw")

    prev_nodes = -1
    it = 0
    max_iter = 10

    while prev_nodes != len(model.graph.node) and it < max_iter:
        prev_nodes = len(model.graph.node)
        it += 1

        # 关键：先把 transpose 从 fork 前面推到 fork 后面
        try:
            model = model.transform(MoveTransposePastFork())
        except Exception as e:
            print("[WARN] MoveTransposePastFork:", e)

        # 再清一下连续 transpose
        model = model.transform(AbsorbConsecutiveTransposes())

        # 然后再把图上的 fork 变成 HW DuplicateStreams
        model = model.transform(to_hw.InferDuplicateStreamsLayer())

        # residual add 这些保留
        model = model.transform(to_hw.InferAddStreamsLayer())

        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        model = model.transform(InferDataLayouts())

    dbg_dir = cfg.output_dir + "/intermediate_models"
    os.makedirs(dbg_dir, exist_ok=True)
    dbg_path = dbg_dir + "/step_after_convert_to_hw_cleanup.onnx"
    model.save(dbg_path)
    print("[PATCH] saved post-convert cleanup checkpoint to:", dbg_path)

    return model

    
def step_convert_to_hw_patched(model, cfg):
    if cfg.standalone_thresholds:
        model = model.transform(to_hw.InferThresholdingLayer())

    model = model.transform(to_hw.InferBinaryMatrixVectorActivation())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())

    # 关键补充：depthwise MatMul -> VVAU
    model = model.transform(to_hw.InferVectorVectorActivation())

    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(to_hw.InferThresholdingLayer())

    need_conv = len(model.get_nodes_by_op_type("Im2Col")) > 0
    if need_conv:
        model = model.transform(to_hw.InferConvInpGen())

    model = model.transform(to_hw.InferStreamingMaxPool())
    model = model.transform(AbsorbConsecutiveTransposes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferDataLayouts())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


BASE_DIR = Path("/home/slowman/Desktop/project/Thesis/finn/notebooks/pytorch-tcn")
MODEL_FILE = BASE_DIR / "onnx_exports" / "ecg5000_static_qtcn_finn_streamlined.onnx"
OUT_DIR = BASE_DIR / "finn_build_pynqz2"

# custom_steps = [
#     "step_streamline",
#     step_after_streamline_cleanup,
#     step_round_thresholds,
#     "step_convert_to_hw",
#     step_after_convert_to_hw_cleanup,
#     "step_create_dataflow_partition",
#     "step_specialize_layers",
#     step_fix_names_after_specialize,
#     "step_target_fps_parallelization",
#     "step_apply_folding_config",
#     "step_minimize_bit_width",
#     "step_generate_estimate_reports",
#     "step_hw_codegen",
#     "step_hw_ipgen",
#     "step_set_fifo_depths",
#     "step_create_stitched_ip",
#     "step_synthesize_bitfile",
#     "step_make_pynq_driver",
#     "step_deployment_package",
# ]

custom_steps = [
    "step_streamline",
    step_after_streamline_cleanup,
    step_round_thresholds,
    #"step_convert_to_hw",
    step_convert_to_hw_patched,
    step_after_convert_to_hw_cleanup,
    "step_create_dataflow_partition",
    "step_specialize_layers",
    step_fix_names_after_specialize,
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    "step_synthesize_bitfile",
    "step_make_pynq_driver",
    "step_deployment_package",
]

# cfg = build_cfg.DataflowBuildConfig(
#     output_dir=str(OUT_DIR),
#     board="Pynq-Z2",
#     shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
#     synth_clk_period_ns=10.0,
#     steps=custom_steps,
#     save_intermediate_models=True,
#     specialize_layers_config_file=str(
#         BASE_DIR / "finn_build_pynqz2" / "template_specialize_layers_config_customized.json"
#     ),
#     generate_outputs=[
#         build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
#         build_cfg.DataflowOutputType.STITCHED_IP,
#         build_cfg.DataflowOutputType.BITFILE,
#         build_cfg.DataflowOutputType.PYNQ_DRIVER,
#         build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
#     ],
# )

cfg = build_cfg.DataflowBuildConfig(
    output_dir=str(OUT_DIR),
    board="Pynq-Z2",
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    synth_clk_period_ns=10.0,
    steps=custom_steps,
    #steps=build_cfg.estimate_only_dataflow_steps,
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


class MoveScalarMulPastConvPipeline(Transformation):
    """x -> Mul(a) -> Pad -> Im2Col -> MatMul -> y
       becomes
       x -> Pad -> Im2Col -> MatMul -> Mul(a) -> y
    """

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        for n in list(graph.node):
            if n.op_type != "Mul":
                continue
            if model.is_fork_node(n) or model.is_join_node(n):
                continue

            const_name, const_val, dyn_name = _split_const_dyn_input(model, n)
            if const_val is None or not _is_scalar_param(const_val):
                continue

            pad = model.find_consumer(n.output[0])
            if pad is None or pad.op_type != "Pad" or model.is_join_node(pad):
                continue

            im2col = model.find_consumer(pad.output[0])
            if im2col is None or im2col.op_type != "Im2Col" or model.is_join_node(im2col):
                continue

            matmul = model.find_consumer(im2col.output[0])
            if matmul is None or matmul.op_type != "MatMul" or model.is_join_node(matmul):
                continue

            if matmul.input[0] != im2col.output[0]:
                continue

            old_mm_out = matmul.output[0]
            mm_out_shape = model.get_tensor_shape(old_mm_out)
            mm_out_layout = model.get_tensor_layout(old_mm_out)

            mid_name = model.make_new_valueinfo_name()
            if mm_out_shape is not None:
                model.set_tensor_shape(mid_name, mm_out_shape)
            if mm_out_layout is not None:
                model.set_tensor_layout(mid_name, mm_out_layout)

            # Mul 从最前面拿掉
            pad.input[0] = dyn_name

            # MatMul 输出改成中间张量
            matmul.output[0] = mid_name

            # 在 MatMul 后面重新插入 Mul
            insert_idx = list(graph.node).index(matmul)
            new_mul = oh.make_node(
                "Mul",
                [mid_name, const_name],
                [old_mm_out],
                name=n.name + "_moved",
            )
            graph.node.insert(insert_idx + 1, new_mul)

            # 删除旧的 Mul
            graph.node.remove(n)
            graph_modified = True

        if graph_modified:
            model = model.transform(SortGraph())
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)