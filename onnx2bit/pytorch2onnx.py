import sys
import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import onnx

from brevitas.export import export_qonnx

from qonnx.util.cleanup import cleanup
from qonnx.core.modelwrapper import ModelWrapper
import qonnx.core.onnx_exec as oxe
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
from qonnx.transformation.infer_shapes import InferShapes

from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    AbsorbMulAddIntoMultiThreshold,
    DuplicateScalarMulAfterFork,
)


# =========================================================
# paths
# 假设本脚本位于 finn/notebooks/onnx2bit/pytorch2onnx.py
# =========================================================

THIS_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = THIS_DIR.parent
PROJECT_DIR = NOTEBOOKS_DIR / "pytorch-tcn"

sys.path.insert(0, str(PROJECT_DIR))

print("THIS_DIR      =", THIS_DIR)
print("NOTEBOOKS_DIR =", NOTEBOOKS_DIR)
print("PROJECT_DIR   =", PROJECT_DIR)

from new_version.newnet import ECG5000FullQuantTCN


# =========================================================
# Checkpoint / export paths
# =========================================================

# 如果你的新模型仍然保存成 best_qtcn_model.pth，就把这里改回去。
CHECKPOINT_PATH = PROJECT_DIR / "new_version" / "best_qtcn_model_last_step.pth"

EXPORT_DIR = PROJECT_DIR / "onnx_exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

RAW_QONNX_PATH = EXPORT_DIR / "ecg5000_pointwise_qtcn_qonnx.onnx"
CLEAN_QONNX_PATH = EXPORT_DIR / "ecg5000_pointwise_qtcn_qonnx_clean.onnx"
CLEAN_QONNX_SHAPED_PATH = EXPORT_DIR / "ecg5000_pointwise_qtcn_qonnx_clean_shaped.onnx"
CLEAN_QONNX_SHAPED_FIXED_PATH = EXPORT_DIR / "ecg5000_pointwise_qtcn_qonnx_clean_shaped_fixed.onnx"
FINN_ONNX_PATH = EXPORT_DIR / "ecg5000_pointwise_qtcn_finn.onnx"
FINN_ONNX_STREAMLINED_PATH = EXPORT_DIR / "ecg5000_pointwise_qtcn_finn_streamlined.onnx"


# =========================================================
# fixed config for static deployment model
# =========================================================

BATCH_SIZE = 1
IN_CH = 1
SEQ_LEN = 140
NUM_CLASSES = 5
DEVICE = "cpu"


# =========================================================
# common helpers
# =========================================================

def qt_value(x):
    return x.value if hasattr(x, "value") else x


def logits_to_2d_torch(x):
    """
    Convert PyTorch model output to [N, C] for accuracy comparison.

    B方案 temporal classifier 的目标输出:
        [N, 5, 1, 1]

    训练/验证/比较时转成:
        [N, 5]
    """
    x = qt_value(x)

    if x.dim() == 2:
        return x

    if x.dim() == 4:
        # Desired B-scheme output: [N, C, 1, 1]
        if x.shape[1] == NUM_CLASSES and x.shape[2] == 1 and x.shape[3] == 1:
            return x[:, :, 0, 0]

        # Fallback old pointwise style: [N, C, L, 1]
        # This should NOT happen for B scheme, but keep it for debugging.
        if x.shape[1] == NUM_CLASSES and x.shape[3] == 1:
            print("[warn] PyTorch model still outputs all time steps [N, C, L, 1].")
            return x[:, :, -1, 0]

        # Possible NHWC style: [N, L, 1, C]
        if x.shape[-1] == NUM_CLASSES:
            print("[warn] PyTorch model output looks NHWC-like and still has time dimension.")
            return x[:, -1, 0, :]

    raise ValueError(f"Expected logits convertible to [N, C], got {tuple(x.shape)}")


def logits_to_2d_np(x):
    """
    Convert ONNX output to [N, C] for comparison.

    B方案目标:
        [N, 5, 1, 1] -> [N, 5]

    旧输出:
        [N, 5, 140, 1] -> [N, 5] using last time step
    """
    x = np.asarray(x)

    if x.ndim == 2:
        return x

    if x.ndim == 4:
        # Desired B-scheme output: [N, C, 1, 1]
        if x.shape[1] == NUM_CLASSES and x.shape[2] == 1 and x.shape[3] == 1:
            return x[:, :, 0, 0]

        # Fallback old style: [N, C, L, 1]
        if x.shape[1] == NUM_CLASSES and x.shape[3] == 1:
            print("[warn] ONNX model still outputs all time steps [N, C, L, 1].")
            return x[:, :, -1, 0]

        # NHWC-like: [N, L, 1, C]
        if x.shape[-1] == NUM_CLASSES and x.shape[2] == 1:
            print("[warn] ONNX output looks NHWC-like and still has time dimension.")
            return x[:, -1, 0, :]

        # Another possible layout: [N, 1, L, C]
        if x.shape[-1] == NUM_CLASSES and x.shape[1] == 1:
            print("[warn] ONNX output looks [N, 1, L, C] and still has time dimension.")
            return x[:, 0, -1, :]

    raise ValueError(f"Expected logits convertible to [N, C], got {x.shape}")


def pick_logits_output_from_onnx_outputs(output_dict, target_num_classes, tag=""):
    """
    Pick an output that looks like classification logits.

    Supported output shapes:
        [N, C]
        [N, C, 1, 1]
        [N, C, L, 1]
        [N, L, 1, C]
        [N, 1, L, C]

    For verification, logits_to_2d_np() will take the last causal time step.
    """
    print(f"\n=== Available ONNX outputs: {tag} ===")
    for name, val in output_dict.items():
        arr = np.asarray(val)
        print(f"{name}: shape={arr.shape}")

    candidates = []

    for name, val in output_dict.items():
        arr = np.asarray(val)

        # [N, C]
        if arr.ndim == 2 and arr.shape[1] == target_num_classes:
            candidates.append((name, arr))
            continue

        if arr.ndim == 4:
            # [N, C, 1, 1]
            if (
                arr.shape[1] == target_num_classes
                and arr.shape[2] == 1
                and arr.shape[3] == 1
            ):
                candidates.append((name, arr))
                continue

            # [N, C, L, 1]
            if arr.shape[1] == target_num_classes and arr.shape[3] == 1:
                candidates.append((name, arr))
                continue

            # [N, L, 1, C]
            if arr.shape[-1] == target_num_classes and arr.shape[2] == 1:
                candidates.append((name, arr))
                continue

            # [N, 1, L, C]
            if arr.shape[-1] == target_num_classes and arr.shape[1] == 1:
                candidates.append((name, arr))
                continue

    if len(candidates) == 0:
        raise RuntimeError(
            f"Could not find logits-like output with class dim = {target_num_classes}. "
            f"See printed output shapes above."
        )

    chosen_name, chosen_val = candidates[0]
    converted = logits_to_2d_np(chosen_val)

    print(f"Chosen logits output: {chosen_name}, raw shape={chosen_val.shape}")
    print(f"Chosen logits converted shape={converted.shape}")

    return chosen_name, chosen_val


# =========================================================
# shape repair helpers
# =========================================================

def extra_streamline_bias_scale_chains(in_path, out_path):
    model = ModelWrapper(str(in_path))

    prev_num_nodes = -1
    while prev_num_nodes != len(model.graph.node):
        prev_num_nodes = len(model.graph.node)

        model = model.transform(DuplicateScalarMulAfterFork())
        model = model.transform(AbsorbMulAddIntoMultiThreshold())
        model = model.transform(AbsorbAddIntoMultiThreshold())
        model = model.transform(AbsorbMulIntoMultiThreshold())
        model = model.transform(AbsorbSignBiasIntoMultiThreshold())

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model.save(str(out_path))
    print(f"Saved extra-streamlined model to: {out_path}")


def _normalize_shape(shape):
    """把 numpy/int 混合的 shape 统一转成 Python int list。"""
    if shape is None:
        return None

    out = []
    for d in shape:
        try:
            out.append(int(d))
        except Exception:
            return None

    return out


def _shape_has_zero(shape):
    return shape is not None and any(int(d) == 0 for d in shape)


def materialize_exec_shapes(onnx_in_path, onnx_out_path, dummy_input_np, verbose=True):
    """
    用一次真实执行，把 clean QONNX 里中间张量的实际 shape 写回去，
    避免后续 ConvertQONNXtoFINN() 用到 [0, ...] 这种坏 shape。

    注意：
    这里故意不再跑 InferShapes()，避免把真实 shape 又推坏成 0。
    """
    model = ModelWrapper(str(onnx_in_path))
    input_name = model.graph.input[0].name

    exec_ctx = oxe.execute_onnx(
        model,
        {input_name: dummy_input_np},
        return_full_exec_context=True,
    )

    updated = 0
    skipped = 0
    repaired = []

    for tensor_name, value in exec_ctx.items():
        if not hasattr(value, "shape"):
            skipped += 1
            continue

        real_shape = _normalize_shape(list(value.shape))

        if real_shape is None or len(real_shape) == 0:
            skipped += 1
            continue

        try:
            old_shape = model.get_tensor_shape(tensor_name)
            old_shape = _normalize_shape(old_shape)
        except Exception:
            old_shape = None

        try:
            model.set_tensor_shape(tensor_name, real_shape)
            updated += 1

            if old_shape != real_shape:
                repaired.append((tensor_name, old_shape, real_shape))

        except Exception:
            skipped += 1

    try:
        model = model.transform(InferDataTypes())
    except Exception as e:
        if verbose:
            print(f"[materialize_exec_shapes] InferDataTypes skipped due to: {e}")

    model.save(str(onnx_out_path))

    if verbose:
        print(f"[materialize_exec_shapes] updated={updated}, skipped={skipped}")
        print(f"[materialize_exec_shapes] repaired_count={len(repaired)}")
        print(f"[materialize_exec_shapes] saved to: {onnx_out_path}")

        if len(repaired) > 0:
            print("\n=== materialized / repaired tensor shapes ===")
            for name, old_shape, new_shape in repaired:
                print(f"{name}: {old_shape} -> {new_shape}")


def print_zero_dim_tensors(onnx_path, tag):
    """
    打印 graph 里所有 shape 含 0 的 tensor。
    """
    model = ModelWrapper(str(onnx_path))
    bad = []

    names = set()

    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        names.add(vi.name)

    for init in model.graph.initializer:
        names.add(init.name)

    for name in sorted(names):
        try:
            shape = model.get_tensor_shape(name)
            shape = _normalize_shape(shape)
        except Exception:
            continue

        if shape is None:
            continue

        if _shape_has_zero(shape):
            bad.append((name, shape))

    print(f"\n=== Zero-dim tensors: {tag} ===")
    if len(bad) == 0:
        print("None")
    else:
        for name, shape in bad:
            print(name, shape)


def replace_zero_dims_with_unknown(onnx_in_path, onnx_out_path):
    """
    把 graph 里 shape 中的 0 改成 unknown，避免某些后续 pass 把 0 当成真实维度。
    """
    model = onnx.load(str(onnx_in_path))

    all_vis = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)

    changed = 0

    for vi in all_vis:
        tt = vi.type.tensor_type

        if not tt.HasField("shape"):
            continue

        dims = tt.shape.dim
        local_changed = False

        for d in dims:
            if d.HasField("dim_value") and d.dim_value == 0:
                d.ClearField("dim_value")
                d.dim_param = "unk"
                local_changed = True

        if local_changed:
            changed += 1

    onnx.save(model, str(onnx_out_path))
    print(f"[replace_zero_dims_with_unknown] changed={changed}, saved to: {onnx_out_path}")


# =========================================================
# model / ONNX inspection helpers
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

        try:
            in_shape = model.get_tensor_shape(in_name)
        except Exception:
            in_shape = None

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
            except Exception:
                attrs[a.name] = "<unparsed>"

        print(f"\n--- MultiThreshold [{i}] ---")
        print("name      :", node.name if node.name else "<unnamed>")
        print("input     :", in_name)
        print("in_shape  :", in_shape)
        print("threshold :", th_name)
        print("th_shape  :", th_shape)
        print("output    :", list(node.output))
        print("attrs     :", attrs)


def build_model_from_checkpoint(ckpt_path: Path):
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "model_state_dict" not in ckpt or "model_config" not in ckpt:
        raise RuntimeError("Checkpoint missing model_state_dict or model_config")

    cfg = ckpt["model_config"]

    model = ECG5000FullQuantTCN(
        num_classes=cfg.get("num_classes", NUM_CLASSES),
        seq_len=cfg.get("seq_len", SEQ_LEN),
        num_inputs=cfg.get("num_inputs", IN_CH),
        num_channels=tuple(cfg.get("num_channels", (16, 16, 16))),
        kernel_size=cfg.get("kernel_size", 5),
        dropout=cfg.get("dropout", 0.0),
        causal=cfg.get("causal", True),
    )

    raw_state_dict = ckpt["model_state_dict"]

    cleaned_state_dict = {}
    for k, v in raw_state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    load_result = model.load_state_dict(cleaned_state_dict, strict=False)

    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)

    print("Checkpoint loaded.")
    print("Model config    :", cfg)
    print("Missing keys    :", missing)
    print("Unexpected keys :", unexpected)

    # 允许 Brevitas 某些内部状态缺失，但不允许真正的参数缺失。
    # 如果你删掉了 export_all_time_logits 这种非参数字段，不会影响这里。
    serious_missing = [
        k for k in missing
        if not k.endswith(".value")
    ]

    serious_unexpected = [
        k for k in unexpected
        if not k.endswith(".value")
    ]

    if len(serious_missing) > 0 or len(serious_unexpected) > 0:
        raise RuntimeError(
            "Checkpoint and current model structure do not match.\n"
            f"Serious missing keys   : {serious_missing}\n"
            f"Serious unexpected keys: {serious_unexpected}"
        )

    return model


def inspect_model(onnx_path: Path, title: str):
    model = ModelWrapper(str(onnx_path))

    print(f"\n=== {title} ===")

    print("Inputs:")
    for x in model.graph.input:
        try:
            shape = model.get_tensor_shape(x.name)
        except Exception:
            shape = None

        try:
            dt = model.get_tensor_datatype(x.name)
        except Exception:
            dt = None

        print(" ", x.name, shape, dt)

    print("Outputs:")
    for x in model.graph.output:
        try:
            shape = model.get_tensor_shape(x.name)
        except Exception:
            shape = None

        try:
            dt = model.get_tensor_datatype(x.name)
        except Exception:
            dt = None

        print(" ", x.name, shape, dt)

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
    cleanup(str(raw_path), out_file=str(clean_path))

    qmodel = ModelWrapper(str(clean_path))
    qmodel = qmodel.transform(InferDataTypes())
    qmodel.save(str(clean_path))

    print(f"Prepared QONNX saved to: {clean_path}")


def verify_pytorch_vs_qonnx(model, qonnx_path: Path, dummy_input):
    model.eval()

    with torch.no_grad():
        pyt_raw_out = model(dummy_input)
        pyt_raw_out = qt_value(pyt_raw_out)
        pyt_out = logits_to_2d_torch(pyt_raw_out).cpu().numpy()

    target_num_classes = pyt_out.shape[1]

    qmodel = ModelWrapper(str(qonnx_path))
    input_name = qmodel.graph.input[0].name

    qonnx_out_dict = oxe.execute_onnx(
        qmodel,
        {input_name: dummy_input.cpu().numpy()},
    )

    _, qonnx_raw_out = pick_logits_output_from_onnx_outputs(
        qonnx_out_dict,
        target_num_classes=target_num_classes,
        tag="QONNX",
    )

    qonnx_out = logits_to_2d_np(qonnx_raw_out)

    diff = np.abs(pyt_out - qonnx_out)
    allclose = np.allclose(pyt_out, qonnx_out, atol=1e-5, rtol=1e-5)

    print("\n=== PyTorch vs QONNX ===")
    print("PyTorch raw output shape :", tuple(pyt_raw_out.shape))
    print("PyTorch logits shape     :", pyt_out.shape)
    print("QONNX raw output shape   :", np.asarray(qonnx_raw_out).shape)
    print("QONNX logits shape       :", qonnx_out.shape)
    print("max abs diff             :", diff.max())
    print("mean abs diff            :", diff.mean())
    print("argmax match             :", (pyt_out.argmax(axis=1) == qonnx_out.argmax(axis=1)).mean())
    print("out_a range              :", pyt_out.min(), pyt_out.max())
    print("out_b range              :", qonnx_out.min(), qonnx_out.max())
    print("Allclose                 :", allclose)

    return allclose, diff.max()


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
            + "\n请确认 QuantReLU 使用的是 unsigned / non-narrow activation quant。"
        )

    print("All Relu -> Quant nodes look compatible.")


def convert_qonnx_to_finn(clean_qonnx_path: Path, finn_onnx_path: Path):
    model = ModelWrapper(str(clean_qonnx_path))

    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    try:
        model = model.transform(InferDataTypes())
    except Exception as e:
        print(f"[WARN] InferDataTypes on FINN model skipped: {e}")

    model.save(str(finn_onnx_path))

    print(f"\nFINN-ONNX saved to: {finn_onnx_path}")


def verify_qonnx_vs_finn(clean_qonnx_path: Path, finn_onnx_path: Path, dummy_input):
    clean_model = ModelWrapper(str(clean_qonnx_path))
    finn_model = ModelWrapper(str(finn_onnx_path))

    input_name_clean = clean_model.graph.input[0].name
    input_name_finn = finn_model.graph.input[0].name

    print("\n=== QONNX(clean) vs FINN-ONNX ===")
    print("CLEAN_QONNX_PATH =", clean_qonnx_path)
    print("FINN_ONNX_PATH   =", finn_onnx_path)

    print("exists CLEAN:", os.path.exists(clean_qonnx_path))
    print("exists FINN :", os.path.exists(finn_onnx_path))
    print("mtime CLEAN :", os.path.getmtime(clean_qonnx_path))
    print("mtime FINN  :", os.path.getmtime(finn_onnx_path))

    out_clean_dict = oxe.execute_onnx(
        clean_model,
        {input_name_clean: dummy_input.cpu().numpy()},
    )

    out_finn_dict = oxe.execute_onnx(
        finn_model,
        {input_name_finn: dummy_input.cpu().numpy()},
    )

    _, out_clean_raw = pick_logits_output_from_onnx_outputs(
        out_clean_dict,
        target_num_classes=NUM_CLASSES,
        tag="CLEAN_QONNX",
    )

    _, out_finn_raw = pick_logits_output_from_onnx_outputs(
        out_finn_dict,
        target_num_classes=NUM_CLASSES,
        tag="FINN_ONNX",
    )

    out_clean = logits_to_2d_np(out_clean_raw)
    out_finn = logits_to_2d_np(out_finn_raw)

    diff = np.abs(out_clean - out_finn)
    allclose = np.allclose(out_clean, out_finn, atol=1e-5, rtol=1e-5)

    print("Clean raw output shape :", np.asarray(out_clean_raw).shape)
    print("FINN raw output shape  :", np.asarray(out_finn_raw).shape)
    print("Clean logits shape     :", out_clean.shape)
    print("FINN logits shape      :", out_finn.shape)
    print("Max abs diff           :", diff.max())
    print("Mean abs diff          :", diff.mean())
    print("Argmax match           :", (out_clean.argmax(axis=1) == out_finn.argmax(axis=1)).mean())
    print("Allclose               :", allclose)

    return allclose, diff.max()


# =========================================================
# Debug helpers
# =========================================================

def debug_tensor_info(onnx_path: Path, tensor_names, title: str):
    """
    打印指定 tensor 的 datatype / shape / producer / consumers
    """
    model = ModelWrapper(str(onnx_path))

    print(f"\n=== Tensor debug: {title} ===")
    print("ONNX:", onnx_path)

    for tname in tensor_names:
        print(f"\n[tensor] {tname}")

        try:
            dt = model.get_tensor_datatype(tname)
        except Exception as e:
            dt = f"<error: {e}>"

        try:
            shape = model.get_tensor_shape(tname)
        except Exception as e:
            shape = f"<error: {e}>"

        print("  dtype :", dt)
        print("  shape :", shape)

        try:
            prod = model.find_producer(tname)
        except Exception as e:
            prod = f"<error: {e}>"

        if prod is None:
            print("  producer: None")
        elif isinstance(prod, str):
            print("  producer:", prod)
        else:
            print("  producer:")
            print("    name   :", prod.name if prod.name else "<unnamed>")
            print("    op_type:", prod.op_type)
            print("    domain :", prod.domain if hasattr(prod, "domain") else "")

        try:
            consumers = model.find_consumers(tname)
        except Exception as e:
            consumers = f"<error: {e}>"

        if consumers is None:
            print("  consumers: None")
        elif isinstance(consumers, str):
            print("  consumers:", consumers)
        else:
            if len(consumers) == 0:
                print("  consumers: []")
            else:
                print("  consumers:")
                for c in consumers:
                    print(
                        "    -",
                        c.name if c.name else "<unnamed>",
                        "|",
                        c.op_type,
                        "|",
                        c.domain if hasattr(c, "domain") else "",
                    )


def inspect_quant_like_nodes(onnx_path: Path, title: str):
    """
    检查图里所有可能和量化相关的节点
    """
    model = ModelWrapper(str(onnx_path))

    quant_like_ops = [
        "Quant",
        "BinaryQuant",
        "Trunc",
        "IntQuant",
        "BipolarQuant",
        "FloatQuant",
        "MultiThreshold",
    ]

    print(f"\n=== Quant-like node inspection: {title} ===")
    print("ONNX:", onnx_path)

    found_any = False

    for op_name in quant_like_ops:
        nodes = [n for n in model.graph.node if n.op_type == op_name]
        print(f"{op_name}: {len(nodes)}")

        if len(nodes) > 0:
            found_any = True

            for i, n in enumerate(nodes[:10]):
                print(f"  [{i}] name={n.name if n.name else '<unnamed>'}")
                print(f"      inputs ={list(n.input)}")
                print(f"      outputs={list(n.output)}")
                if hasattr(n, "domain"):
                    print(f"      domain ={n.domain}")

    if not found_any:
        print("No quant-like nodes found.")


def inspect_first_n_nodes(onnx_path: Path, n=30, title=""):
    """
    打印图开头若干个节点，方便看 input_quant 有没有真的导出来
    """
    model = ModelWrapper(str(onnx_path))

    print(f"\n=== First {n} nodes: {title} ===")
    print("ONNX:", onnx_path)

    for i, node in enumerate(model.graph.node[:n]):
        print(f"\n[{i}]")
        print("  name   :", node.name if node.name else "<unnamed>")
        print("  op_type:", node.op_type)
        print("  domain :", node.domain if hasattr(node, "domain") else "")
        print("  inputs :", list(node.input))
        print("  outputs:", list(node.output))


def find_tensors_by_keyword(onnx_path: Path, keywords, title: str):
    """
    在 graph 里搜带关键字的 tensor 名
    """
    model = ModelWrapper(str(onnx_path))

    all_names = set()

    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        all_names.add(vi.name)

    for init in model.graph.initializer:
        all_names.add(init.name)

    for node in model.graph.node:
        for x in node.input:
            all_names.add(x)
        for x in node.output:
            all_names.add(x)

    print(f"\n=== Tensor name search: {title} ===")
    print("keywords =", keywords)

    matched = []

    for name in sorted(all_names):
        low = name.lower()
        if any(k.lower() in low for k in keywords):
            matched.append(name)

    if len(matched) == 0:
        print("No matched tensor names.")
    else:
        for name in matched:
            print(name)

    return matched


def find_nodes_by_keyword(onnx_path: Path, keywords, title: str):
    """
    在 graph 里搜带关键字的 node 名 / op_type
    """
    model = ModelWrapper(str(onnx_path))

    print(f"\n=== Node search: {title} ===")
    print("keywords =", keywords)

    found = 0

    for i, node in enumerate(model.graph.node):
        nname = node.name if node.name else ""
        opname = node.op_type if node.op_type else ""
        domain = node.domain if hasattr(node, "domain") else ""

        target = f"{nname} {opname} {domain}".lower()

        if any(k.lower() in target for k in keywords):
            found += 1

            print(f"\n[{i}]")
            print("  name   :", nname if nname else "<unnamed>")
            print("  op_type:", opname)
            print("  domain :", domain)
            print("  inputs :", list(node.input))
            print("  outputs:", list(node.output))

    if found == 0:
        print("No matched nodes.")


def debug_raw_clean_finn_all():
    """
    一次性检查 RAW / CLEAN / FINN 三个文件
    """
    paths = [
        ("RAW_QONNX", RAW_QONNX_PATH),
        ("CLEAN_QONNX", CLEAN_QONNX_PATH),
        ("CLEAN_QONNX_SHAPED_FIXED", CLEAN_QONNX_SHAPED_FIXED_PATH),
        ("FINN_ONNX", FINN_ONNX_PATH),
        ("FINN_ONNX_STREAMLINED", FINN_ONNX_STREAMLINED_PATH),
    ]

    target_tensors = [
        "global_in",
        "input_quant_out",
        "Im2Col_0_out0",
        "MatMul_1_out0",
    ]

    for tag, path in paths:
        if not path.exists():
            print(f"\n=== {tag} not found: {path} ===")
            continue

        print(f"\n\n############################")
        print(f"### DEBUG FILE: {tag}")
        print(f"### PATH: {path}")
        print(f"############################")

        inspect_model(path, f"{tag} / op histogram")
        inspect_quant_like_nodes(path, f"{tag} / quant-like ops")
        inspect_first_n_nodes(path, n=25, title=f"{tag} / first nodes")
        find_tensors_by_keyword(path, ["quant", "input", "im2col", "matmul"], f"{tag} / tensor search")
        find_nodes_by_keyword(path, ["quant", "im2col", "matmul", "multithreshold"], f"{tag} / node search")
        debug_tensor_info(path, target_tensors, f"{tag} / target tensors")


def inspect_final_classifier_candidate(onnx_path: Path, title: str):
    """
    检查最终 pointwise classifier 是否已经变成 MW=32, MH=5 风格。
    这一步不保证所有图阶段都有 MVAU，只是辅助检查。
    """
    model = ModelWrapper(str(onnx_path))

    print(f"\n=== Final classifier candidate inspection: {title} ===")
    print("ONNX:", onnx_path)

    for n in model.graph.node:
        if "MVAU" not in n.op_type and "MatrixVector" not in n.op_type:
            continue

        try:
            cop = getCustomOp(n)
        except Exception:
            continue

        attrs = {}
        for a in ["MW", "MH", "SIMD", "PE", "inputDataType", "weightDataType", "outputDataType", "numInputVectors", "noActivation"]:
            try:
                attrs[a] = cop.get_nodeattr(a)
            except Exception:
                pass

        if attrs.get("MH", None) == NUM_CLASSES:
            print("\nnode:", n.name if n.name else "<unnamed>")
            print("op_type:", n.op_type)
            print("domain:", n.domain)
            print("inputs:", list(n.input))
            print("outputs:", list(n.output))
            print("attrs:", attrs)

            for t in list(n.input) + list(n.output):
                try:
                    print("  tensor", t, "shape", model.get_tensor_shape(t), "dtype", model.get_tensor_datatype(t))
                except Exception:
                    pass


# =========================================================
# main
# =========================================================

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print("CHECKPOINT :", CHECKPOINT_PATH)
    print("EXPORT_DIR :", EXPORT_DIR)

    model = build_model_from_checkpoint(CHECKPOINT_PATH).to(DEVICE)
    model.eval()

    dummy_input = torch.randn(BATCH_SIZE, IN_CH, SEQ_LEN, 1, device=DEVICE)
    dummy_input_np = dummy_input.detach().cpu().numpy()

    with torch.no_grad():
        pyt_raw_out = model(dummy_input)
        pyt_raw_out = qt_value(pyt_raw_out)
        pyt_logits = logits_to_2d_torch(pyt_raw_out)

    print("\n=== PyTorch raw output before export ===")
    print("raw output shape       :", tuple(pyt_raw_out.shape))
    print("converted logits shape :", tuple(pyt_logits.shape))
    print("raw output range       :", float(pyt_raw_out.min()), float(pyt_raw_out.max()))
    print("last-step logits range :", float(pyt_logits.min()), float(pyt_logits.max()))
    print("last-step pred         :", pyt_logits.argmax(dim=1).detach().cpu().numpy())

    expected_raw_shape = (BATCH_SIZE, NUM_CLASSES, 1, 1)
    if tuple(pyt_raw_out.shape) != expected_raw_shape:
        raise RuntimeError(
            "B-scheme model output shape is wrong before export.\n"
            f"Expected PyTorch raw output shape: {expected_raw_shape}\n"
            f"Got: {tuple(pyt_raw_out.shape)}\n"
            "This means your newnet.py is still not using the temporal classifier "
            "that directly outputs [N, 5, 1, 1], or the checkpoint/model class is not the B version."
        )

    print("\n=== Exporting raw QONNX ===")
    export_model_to_qonnx(model, dummy_input, RAW_QONNX_PATH)
    inspect_model(RAW_QONNX_PATH, "Raw QONNX")

    # ===== DEBUG RAW QONNX =====
    inspect_quant_like_nodes(RAW_QONNX_PATH, "RAW_QONNX")
    inspect_first_n_nodes(RAW_QONNX_PATH, n=25, title="RAW_QONNX")
    find_tensors_by_keyword(RAW_QONNX_PATH, ["quant", "input", "im2col", "matmul", "classifier"], "RAW_QONNX")
    find_nodes_by_keyword(RAW_QONNX_PATH, ["quant", "im2col", "matmul", "multithreshold", "classifier"], "RAW_QONNX")
    debug_tensor_info(
        RAW_QONNX_PATH,
        ["global_in", "input_quant_out", "Im2Col_0_out0", "MatMul_1_out0"],
        "RAW_QONNX",
    )

    print("\n=== Preparing QONNX ===")
    prepare_qonnx(RAW_QONNX_PATH, CLEAN_QONNX_PATH)
    inspect_model(CLEAN_QONNX_PATH, "Prepared QONNX")

    print_zero_dim_tensors(CLEAN_QONNX_PATH, "before materialize")

    materialize_exec_shapes(
        CLEAN_QONNX_PATH,
        CLEAN_QONNX_SHAPED_PATH,
        dummy_input_np,
        verbose=True,
    )

    inspect_model(CLEAN_QONNX_SHAPED_PATH, "Prepared QONNX + materialized shapes")
    print_zero_dim_tensors(CLEAN_QONNX_SHAPED_PATH, "after materialize")

    replace_zero_dims_with_unknown(
        CLEAN_QONNX_SHAPED_PATH,
        CLEAN_QONNX_SHAPED_FIXED_PATH,
    )

    inspect_model(CLEAN_QONNX_SHAPED_FIXED_PATH, "Prepared QONNX + materialized + zero-fixed")
    print_zero_dim_tensors(CLEAN_QONNX_SHAPED_FIXED_PATH, "after replace_zero_dims_with_unknown")

    print("\n=== Verifying PyTorch vs QONNX ===")
    verify_pytorch_vs_qonnx(model, CLEAN_QONNX_SHAPED_FIXED_PATH, dummy_input)

    print("\n=== Checking FINN compatibility ===")
    check_relu_quant_compatibility(CLEAN_QONNX_SHAPED_FIXED_PATH)

    print("\n=== Converting QONNX to FINN-ONNX ===")
    convert_qonnx_to_finn(CLEAN_QONNX_SHAPED_FIXED_PATH, FINN_ONNX_PATH)
    inspect_model(FINN_ONNX_PATH, "FINN-ONNX")
    inspect_final_classifier_candidate(FINN_ONNX_PATH, "FINN-ONNX before extra streamline")

    print("\n=== Extra streamline bias/scale chains ===")
    extra_streamline_bias_scale_chains(FINN_ONNX_PATH, FINN_ONNX_STREAMLINED_PATH)

    inspect_model(FINN_ONNX_STREAMLINED_PATH, "FINN-ONNX STREAMLINED")
    print_zero_dim_tensors(FINN_ONNX_STREAMLINED_PATH, "FINN-ONNX STREAMLINED")
    inspect_multithreshold_nodes(FINN_ONNX_STREAMLINED_PATH, "FINN-ONNX STREAMLINED")
    inspect_final_classifier_candidate(FINN_ONNX_STREAMLINED_PATH, "FINN-ONNX STREAMLINED")

    print("\n=== Verifying QONNX vs FINN-ONNX STREAMLINED ===")
    verify_qonnx_vs_finn(CLEAN_QONNX_SHAPED_FIXED_PATH, FINN_ONNX_STREAMLINED_PATH, dummy_input)

    print("\nDone.")
    print(f"Raw QONNX              : {RAW_QONNX_PATH}")
    print(f"Clean QONNX            : {CLEAN_QONNX_PATH}")
    print(f"Clean QONNX shaped     : {CLEAN_QONNX_SHAPED_PATH}")
    print(f"Clean QONNX shaped fix : {CLEAN_QONNX_SHAPED_FIXED_PATH}")
    print(f"FINN-ONNX              : {FINN_ONNX_PATH}")
    print(f"FINN-ONNX streamlined  : {FINN_ONNX_STREAMLINED_PATH}")

    print("\nExpected B-scheme behavior:")
    print("  PyTorch / QONNX raw output should be [1, 5, 1, 1].")
    print("  Verification directly uses logits[:, :, 0, 0].")
    print("  The final classifier is temporal: kernel_size=(140, 1).")
    print("  In FINN, the final classifier may become MW=32*140=4480, MH=5.")
    print("  This is expected for B scheme and means the time dimension is reduced inside the classifier.")


if __name__ == "__main__":
    main()