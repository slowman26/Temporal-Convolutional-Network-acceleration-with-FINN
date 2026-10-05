from pathlib import Path
import json

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

from finn.builder.build_dataflow_steps import (
    step_specialize_layers,
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
    step_create_stitched_ip,
    step_synthesize_bitfile,
    step_make_pynq_driver,
    step_deployment_package,
)

from qonnx.custom_op.registry import getCustomOp


# ============================================================
# Paths
# ============================================================

OLD_BUILD_DIR = Path("/home/slowman/Desktop/project/Thesis/finn/notebooks/pytorch-tcn/finn_build_pynqz2")
INPUT_MODEL = OLD_BUILD_DIR / "intermediate_models" / "step_target_fps_parallelization.onnx"

NEW_BUILD_DIR = Path("/home/slowman/Desktop/project/Thesis/finn/notebooks/pytorch-tcn/finn_build_pynqz2_force_final_hls")

assert INPUT_MODEL.exists(), f"Missing input model: {INPUT_MODEL}"


# ============================================================
# Custom step: force final MVAU to HLS
# ============================================================

def step_force_final_mvau_hls(model, cfg):
    target = None

    for n in model.graph.node:
        if n.op_type == "MVAU":
            for out in n.output:
                try:
                    shp = model.get_tensor_shape(out)
                    if shp == [1, 1, 1, 5]:
                        target = n
                except Exception:
                    pass

    if target is None:
        raise RuntimeError("Could not find final MVAU with output shape [1, 1, 1, 5].")

    print("Forcing final MVAU to HLS:", target.name)

    op = getCustomOp(target)

    # Conservative folding for debug
    op.set_nodeattr("PE", 1)
    op.set_nodeattr("SIMD", 1)

    # Key setting: avoid RTL MVAU path
    op.set_nodeattr("mem_mode", "internal_embedded")

    # Keep final output as raw accumulator, no threshold activation
    try:
        op.set_nodeattr("noActivation", 1)
    except Exception:
        pass

    return model


# ============================================================
# Build config
# ============================================================

gen_outputs = []
for name in ["ESTIMATE_REPORTS", "BITFILE", "PYNQ_DRIVER", "DEPLOYMENT_PACKAGE"]:
    if hasattr(build_cfg.DataflowOutputType, name):
        gen_outputs.append(getattr(build_cfg.DataflowOutputType, name))

cfg = build_cfg.DataflowBuildConfig(
    output_dir=str(NEW_BUILD_DIR),
    synth_clk_period_ns=10.0,
    board="Pynq-Z2",
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    generate_outputs=gen_outputs,

    # Do not use the old mismatched folding config
    folding_config_file=None,

    steps=[
        step_force_final_mvau_hls,
        step_specialize_layers,
        step_hw_codegen,
        step_hw_ipgen,
        step_set_fifo_depths,
        step_create_stitched_ip,
        step_synthesize_bitfile,
        step_make_pynq_driver,
        step_deployment_package,
    ],
)

build.build_dataflow_cfg(str(INPUT_MODEL), cfg)