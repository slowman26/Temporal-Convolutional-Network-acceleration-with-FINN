import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch

from brevitas.export import export_qonnx

from qonnx.core.modelwrapper import ModelWrapper
import qonnx.core.onnx_exec as oxe
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes

from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

import onnx
import os

from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
#from qonnx.transformation.infer_shapes import InferShapes
#from qonnx.transformation.infer_datatypes import InferDataTypes

# =========================================================
# paths
# 假设本脚本位于 pytorch-tcn/tests/export_finn.py
# =========================================================
THIS_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = THIS_DIR.parent
PROJECT_DIR = NOTEBOOKS_DIR / "pytorch-tcn"

sys.path.insert(0, str(PROJECT_DIR))

print("THIS_DIR      =", THIS_DIR)
print("NOTEBOOKS_DIR =", NOTEBOOKS_DIR)
print("PROJECT_DIR   =", PROJECT_DIR)

from solid_net.model_factory import build_export_model


CHECKPOINT_PATH = PROJECT_DIR / "tests" / "best_qtcn_model.pth"
EXPORT_DIR = PROJECT_DIR / "onnx_exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

RAW_QONNX_PATH = EXPORT_DIR / "ecg5000_static_qtcn_qonnx.onnx"
CLEAN_QONNX_PATH = EXPORT_DIR / "ecg5000_static_qtcn_qonnx_clean.onnx"
FINN_ONNX_PATH = EXPORT_DIR / "ecg5000_static_qtcn_finn.onnx"


# =========================================================
# fixed config for static deployment model
# =========================================================
BATCH_SIZE = 1
IN_CH = 1
SEQ_LEN = 140
NUM_CLASSES = 5
DEVICE = "cpu"


# =========================================================
# model
# =========================================================

def inspect_multithreshold_nodes(onnx_path: Path, title: str):
    model = ModelWrapper(str(onnx_path))

    print(f"\n=== Inspect MultiThreshold: {title} ===")
    mt_nodes = [n for n in model.graph.node if n.op_type == "MultiThreshold"]
    print("MultiThreshold count:", len(mt_nodes))

    if len(mt_nodes) == 0:
        print("No MultiThreshold nodes found.")
        return

    for i, node in enumerate(mt_nodes):
        in_name = node.input[0]
        th_name = node.input[1]

        in_shape = model.get_tensor_shape(in_name)
        th = model.get_initializer(th_name)
        th_shape = None if th is None else th.shape

        attrs = {}
        for a in node.attribute:
            try:
                if a.type == 3:   # STRING
                    attrs[a.name] = a.s.decode("utf-8")
                elif a.type == 2: # INT
                    attrs[a.name] = a.i
                elif a.type == 1: # FLOAT
                    attrs[a.name] = a.f
                else:
                    attrs[a.name] = f"<type {a.type}>"
            except:
                attrs[a.name] = "<unparsed>"

        print(f"\n--- MultiThreshold [{i}] ---")
        print("name      :", node.name if node.name else "<unnamed>")
        print("input     :", in_name)
        print("in_shape  :", in_shape)
        print("threshold :", th_name)
        print("th_shape  :", th_shape)
        print("output    :", list(node.output))
        print("attrs     :", attrs)

def build_model():
    """
    这里必须和你“当前实际训练所用模型结构”一致。
    如果你已经切到 quant_tcn_solid + causal=True 的版本，
    那这里也必须保持一致。
    """
    model = build_export_model(
        num_classes=5,
        seq_len=140,
        num_inputs=1,
        num_channels=(16, 16, 16),
        kernel_size=3,
        dropout=0.1,
        causal=True,
        use_skip_connections=False,
        input_shape="NCL",
    )
    return model


# =========================================================
# utils
# =========================================================

def _fix_zero_dims_in_valueinfo_list(valueinfo_list):
    for vi in valueinfo_list:
        if not vi.type.HasField("tensor_type"):
            continue
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            continue
        for i, dim in enumerate(tt.shape.dim):
            if dim.HasField("dim_value") and dim.dim_value == 0:
                dim.ClearField("dim_value")
                dim.dim_param = f"unk_{vi.name}_{i}"


def sanitize_zero_dims_as_unknown(onnx_path: str):
    model = onnx.load(onnx_path)

    _fix_zero_dims_in_valueinfo_list(model.graph.input)
    _fix_zero_dims_in_valueinfo_list(model.graph.output)
    _fix_zero_dims_in_valueinfo_list(model.graph.value_info)

    onnx.save(model, onnx_path)
    print(f"Sanitized zero dims -> unknown in: {onnx_path}")

def load_checkpoint(model, ckpt_path: Path):
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format.")

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    print("Checkpoint loaded.")
    print("Missing keys   :", missing)
    print("Unexpected keys:", unexpected)

    if len(missing) > 0 or len(unexpected) > 0:
        print("\n[Warning] checkpoint 和当前模型结构没有完全匹配。")
        print("如果你刚把网络改成了新的 static / solid 版本，这通常说明你需要用新模型重新训练一次。")

    return model


def inspect_model(onnx_path: Path, title: str):
    model = ModelWrapper(str(onnx_path))

    print(f"\n=== {title} ===")
    print("Inputs:")
    for x in model.graph.input:
        print(" ", x.name, model.get_tensor_shape(x.name))

    print("Outputs:")
    for x in model.graph.output:
        print(" ", x.name, model.get_tensor_shape(x.name))

    ops = Counter([n.op_type for n in model.graph.node])
    print("Op histogram:")
    for k, v in sorted(ops.items()):
        print(f"  {k}: {v}")


def export_model_to_qonnx(model, dummy_input, out_path: Path):
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)

    export_qonnx(
        model,
        dummy_input,
        str(out_path),
    )
    print(f"\nRaw QONNX saved to: {out_path}")


def prepare_qonnx(raw_path: Path, clean_path: Path):
    qmodel = ModelWrapper(str(raw_path))
    qmodel = qmodel.transform(InferShapes())
    qmodel = qmodel.transform(InferDataTypes())
    qmodel.save(str(clean_path))
    print(f"Prepared QONNX saved to: {clean_path}")

def verify_pytorch_vs_qonnx(model, qonnx_path: Path, dummy_input):
    model.eval()
    with torch.no_grad():
        pyt_out = model(dummy_input).detach().cpu().numpy()

    qmodel = ModelWrapper(str(qonnx_path))
    qmodel = qmodel.transform(InferShapes())

    input_name = qmodel.graph.input[0].name
    output_name = qmodel.graph.output[0].name


    
    qonnx_out_dict = oxe.execute_onnx(
        qmodel,
        {input_name: dummy_input.cpu().numpy()},
    )
    qonnx_out = qonnx_out_dict[output_name]

    max_abs_diff = np.max(np.abs(pyt_out - qonnx_out))
    allclose = np.allclose(pyt_out, qonnx_out, atol=1e-5, rtol=1e-5)

    print("\n=== PyTorch vs QONNX ===")
    print("PyTorch output shape :", pyt_out.shape)
    print("QONNX output shape   :", qonnx_out.shape)
    print("Max abs diff         :", max_abs_diff)
    print("Allclose             :", allclose)

    return allclose, max_abs_diff


def check_relu_quant_compatibility(qonnx_path: Path):
    """
    检查 Relu -> Quant 是否满足 FINN 要求：
    通常 QuantReLU 应该是 unsigned / non-narrow。
    """
    qmodel = ModelWrapper(str(qonnx_path))
    bad_nodes = []

    print("\n=== Checking Relu -> Quant compatibility for FINN ===")
    for node in qmodel.graph.node:
        if node.op_type != "Quant":
            continue

        preds = qmodel.find_direct_predecessors(node)
        if preds is None or len(preds) == 0:
            continue

        prev = preds[0]
        if prev.op_type != "Relu":
            continue

        qop = getCustomOp(node)
        signed = qop.get_nodeattr("signed")
        narrow = qop.get_nodeattr("narrow")

        print(f"Node: {node.name if node.name else '<unnamed>'}")
        print(f"  signed = {signed}")
        print(f"  narrow = {narrow}")

        if signed or narrow:
            bad_nodes.append(node.name if node.name else "<unnamed>")

    if len(bad_nodes) > 0:
        raise RuntimeError(
            "Found FINN-incompatible Relu->Quant node(s): "
            + ", ".join(bad_nodes)
            + "\n请确认 QuantReLU 使用的是 Uint8ActPerTensorFloat。"
        )

    print("All Relu -> Quant nodes look compatible.")


def convert_qonnx_to_finn(clean_qonnx_path: Path, finn_onnx_path: Path):
    model = ModelWrapper(str(clean_qonnx_path))
    model = model.transform(ConvertQONNXtoFINN())

    #====================================
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    for i, n in enumerate(model.graph.node):
        if n.name is None or n.name == "":
            print("EMPTY NODE NAME:", i, n.op_type, n.domain, list(n.input), list(n.output))
    
    #====================================
    model.save(str(finn_onnx_path))
    print(f"\nFINN-ONNX saved to: {finn_onnx_path}")


def verify_qonnx_vs_finn(clean_qonnx_path: Path, finn_onnx_path: Path, dummy_input):
    clean_model = ModelWrapper(str(clean_qonnx_path))
    finn_model = ModelWrapper(str(finn_onnx_path))

    input_name_clean = clean_model.graph.input[0].name
    output_name_clean = clean_model.graph.output[0].name

    input_name_finn = finn_model.graph.input[0].name
    output_name_finn = finn_model.graph.output[0].name

    print("\n=== QONNX(clean) vs FINN-ONNX ===")
    #print("Allclose     :", allclose)

    print("CLEAN_QONNX_PATH =", CLEAN_QONNX_PATH)
    print("FINN_ONNX_PATH   =", FINN_ONNX_PATH)

    
    print("exists CLEAN:", os.path.exists(CLEAN_QONNX_PATH))
    print("exists FINN :", os.path.exists(FINN_ONNX_PATH))
    print("mtime CLEAN :", os.path.getmtime(CLEAN_QONNX_PATH))
    print("mtime FINN  :", os.path.getmtime(FINN_ONNX_PATH))
    
    out_clean_dict = oxe.execute_onnx(
        clean_model,
        {input_name_clean: dummy_input.cpu().numpy()},
    )
    out_finn_dict = oxe.execute_onnx(
        finn_model,
        {input_name_finn: dummy_input.cpu().numpy()},
    )

    out_clean = out_clean_dict[output_name_clean]
    out_finn = out_finn_dict[output_name_finn]

    max_abs_diff = np.max(np.abs(out_clean - out_finn))
    allclose = np.allclose(out_clean, out_finn, atol=1e-5, rtol=1e-5)



    return allclose, max_abs_diff


# =========================================================
# main
# =========================================================
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print("CHECKPOINT :", CHECKPOINT_PATH)
    print("EXPORT_DIR :", EXPORT_DIR)

    model = build_model().to(DEVICE)
    model = load_checkpoint(model, CHECKPOINT_PATH)
    model.eval()

    dummy_input = torch.randn(BATCH_SIZE, IN_CH, SEQ_LEN, device=DEVICE)

    print("\n=== Exporting raw QONNX ===")
    export_model_to_qonnx(model, dummy_input, RAW_QONNX_PATH)
    inspect_model(RAW_QONNX_PATH, "Raw QONNX")

    print("\n=== Preparing QONNX ===")
    prepare_qonnx(RAW_QONNX_PATH, CLEAN_QONNX_PATH)

    
    inspect_model(CLEAN_QONNX_PATH, "Prepared QONNX")

    print("\n=== Verifying PyTorch vs QONNX ===")
    print("FINN_ONNX_PATH used by verify =", FINN_ONNX_PATH)
    verify_pytorch_vs_qonnx(model, CLEAN_QONNX_PATH, dummy_input)

    print("\n=== Checking FINN compatibility ===")
    check_relu_quant_compatibility(CLEAN_QONNX_PATH)

    print("\n=== Converting QONNX to FINN-ONNX ===")
    convert_qonnx_to_finn(CLEAN_QONNX_PATH, FINN_ONNX_PATH)
    sanitize_zero_dims_as_unknown(FINN_ONNX_PATH)
    inspect_model(FINN_ONNX_PATH, "FINN-ONNX")
    inspect_multithreshold_nodes(FINN_ONNX_PATH, "FINN-ONNX")


    print("\n=== Verifying QONNX vs FINN-ONNX ===")
    verify_qonnx_vs_finn(CLEAN_QONNX_PATH, FINN_ONNX_PATH, dummy_input)

    print("\nDone.")
    print(f"Raw QONNX   : {RAW_QONNX_PATH}")
    print(f"Clean QONNX : {CLEAN_QONNX_PATH}")
    print(f"FINN-ONNX   : {FINN_ONNX_PATH}")


if __name__ == "__main__":
    main()