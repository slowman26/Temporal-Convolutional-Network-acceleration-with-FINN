from pathlib import Path

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
)


from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

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


BASE_DIR = Path("/home/slowman/Desktop/project/Thesis/finn/notebooks/pytorch-tcn")
MODEL_FILE = BASE_DIR / "onnx_exports" / "ecg5000_static_qtcn_finn.onnx"
OUT_DIR = BASE_DIR / "finn_build_pynqz2"

custom_steps = [
    step_tidy_up_no_fold,
    "step_streamline",
    "step_convert_to_hw",
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

cfg = build_cfg.DataflowBuildConfig(
    output_dir=str(OUT_DIR),
    board="Pynq-Z2",
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    synth_clk_period_ns=10.0,
    steps=custom_steps,
    save_intermediate_models=True,
    specialize_layers_config_file="/home/slowman/Desktop/project/Thesis/finn/notebooks/pytorch-tcn/finn_build_pynqz2/template_specialize_layers_config_customized.json",
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ],
)

build.build_dataflow_cfg(str(MODEL_FILE), cfg)