import sys
import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch

from qonnx.core.modelwrapper import ModelWrapper
import qonnx.core.onnx_exec as oxe
from qonnx.custom_op.registry import getCustomOp


# ============================================================
# Paths
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = THIS_DIR.parent
PROJECT_DIR = NOTEBOOKS_DIR / "pytorch-tcn"

sys.path.insert(0, str(PROJECT_DIR))

from new_version.newnet import ECG5000FullQuantTCN


CHECKPOINT_PATH = PROJECT_DIR / "new_version" / "best_qtcn_model.pth"

TEST_FILE = PROJECT_DIR / "datasets" / "ECG5000" / "ECG5000_TEST.txt"

EXPORT_DIR = PROJECT_DIR / "onnx_exports"

CLEAN_QONNX_PATH = EXPORT_DIR / "ecg5000_pointwise_qtcn_qonnx_clean_shaped_fixed.onnx"
FINN_ONNX_STREAMLINED_PATH = EXPORT_DIR / "ecg5000_pointwise_qtcn_finn_streamlined.onnx"

OUT_DIR = PROJECT_DIR / "verify_pointwise_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Config
# ============================================================

NUM_CLASSES = 5
SEQ_LEN = 140
IN_CH = 1
DEVICE = "cpu"

# None means use all test samples.
# For quick check, use 50 or 100.
MAX_SAMPLES = 100

# If not None, select balanced samples from each class.
# Example: 10 means 10 samples per class, total 50 samples.
# If this is not None, it overrides MAX_SAMPLES.
DEBUG_BALANCED_PER_CLASS = None

ATOL = 1e-4
RTOL = 1e-4


# ============================================================
# Basic helpers
# ============================================================

def qt_value(x):
    return x.value if hasattr(x, "value") else x


def safe_bincount(x, minlength=5):
    x = np.asarray(x).astype(np.int32).reshape(-1)
    x = x[x >= 0]
    if x.size == 0:
        return np.zeros((minlength,), dtype=np.int32)
    return np.bincount(x, minlength=minlength)


def load_ecg5000_txt(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"ECG5000 txt not found: {path}")

    data = np.loadtxt(str(path), dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D ECG5000 txt, got {data.shape}")

    if data.shape[1] != 141:
        raise ValueError(f"Expected 141 columns, got {data.shape[1]}")

    labels_raw = data[:, 0].astype(np.int32)
    x = data[:, 1:].astype(np.float32)

    unique = sorted(np.unique(labels_raw).astype(np.int32).tolist())

    if unique == [1, 2, 3, 4, 5]:
        labels = labels_raw - 1
        print("[info] Converted labels from 1..5 to 0..4")
    elif unique == [0, 1, 2, 3, 4]:
        labels = labels_raw
        print("[info] Labels already 0..4")
    else:
        label_map = {int(v): i for i, v in enumerate(unique)}
        labels = np.asarray([label_map[int(v)] for v in labels_raw], dtype=np.int32)
        print("[warn] Non-standard labels:", unique)
        print("[warn] Label map:", label_map)

    # [N, 140] -> [N, 1, 140, 1]
    x = x.reshape(x.shape[0], 1, SEQ_LEN, 1).astype(np.float32)

    labels = labels.astype(np.int32)

    print("[info] Full test samples:", len(labels))
    print("[info] Full label distribution:", safe_bincount(labels, minlength=NUM_CLASSES))

    return x, labels


def select_subset(x, labels):
    labels = np.asarray(labels).astype(np.int32).reshape(-1)

    if DEBUG_BALANCED_PER_CLASS is not None:
        chosen = []

        for c in range(NUM_CLASSES):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                print(f"[warn] No samples for class {c}")
                continue
            chosen.extend(idx[:DEBUG_BALANCED_PER_CLASS].astype(np.int32).tolist())

        chosen = np.asarray(chosen, dtype=np.int32)

        x_sel = x[chosen]
        y_sel = labels[chosen]

        print("[info] Using balanced debug subset")
        print("[info] selected indices:", chosen)
        print("[info] subset samples:", len(y_sel))
        print("[info] subset label distribution:", safe_bincount(y_sel, minlength=NUM_CLASSES))

        return x_sel, y_sel, chosen

    if MAX_SAMPLES is not None:
        n = int(MAX_SAMPLES)
        x_sel = x[:n]
        y_sel = labels[:n]
        idx = np.arange(n, dtype=np.int32)

        print("[info] Using first MAX_SAMPLES:", n)
        print("[info] subset label distribution:", safe_bincount(y_sel, minlength=NUM_CLASSES))

        return x_sel, y_sel, idx

    idx = np.arange(len(labels), dtype=np.int32)

    print("[info] Using all samples")
    print("[info] subset label distribution:", safe_bincount(labels, minlength=NUM_CLASSES))

    return x, labels, idx


# ============================================================
# Logit shape helpers
# ============================================================

def logits_to_last_step_np(x):
    """
    Convert raw model/ONNX output to [N, 5] by taking last causal time step.

    Supported layouts:
        [N, 5]
        [N, 5, 1, 1]
        [N, 5, L, 1]
        [N, L, 1, 5]
        [N, 1, L, 5]
    """
    x = np.asarray(x)

    if x.ndim == 2:
        if x.shape[1] != NUM_CLASSES:
            raise ValueError(f"Expected [N, {NUM_CLASSES}], got {x.shape}")
        return x

    if x.ndim == 4:
        # [N, C, 1, 1]
        if x.shape[1] == NUM_CLASSES and x.shape[2] == 1 and x.shape[3] == 1:
            return x[:, :, 0, 0]

        # [N, C, L, 1]
        if x.shape[1] == NUM_CLASSES and x.shape[3] == 1:
            return x[:, :, -1, 0]

        # [N, L, 1, C]
        if x.shape[-1] == NUM_CLASSES and x.shape[2] == 1:
            return x[:, -1, 0, :]

        # [N, 1, L, C]
        if x.shape[-1] == NUM_CLASSES and x.shape[1] == 1:
            return x[:, 0, -1, :]

    raise ValueError(f"Unsupported logits shape: {x.shape}")


def logits_to_last_step_torch(x):
    x = qt_value(x)

    if x.dim() == 2:
        if x.shape[1] != NUM_CLASSES:
            raise ValueError(f"Expected [N, {NUM_CLASSES}], got {tuple(x.shape)}")
        return x

    if x.dim() == 4:
        # [N, C, 1, 1]
        if x.shape[1] == NUM_CLASSES and x.shape[2] == 1 and x.shape[3] == 1:
            return x[:, :, 0, 0]

        # [N, C, L, 1]
        if x.shape[1] == NUM_CLASSES and x.shape[3] == 1:
            return x[:, :, -1, 0]

        # [N, L, 1, C]
        if x.shape[-1] == NUM_CLASSES and x.shape[2] == 1:
            return x[:, -1, 0, :]

        # [N, 1, L, C]
        if x.shape[-1] == NUM_CLASSES and x.shape[1] == 1:
            return x[:, 0, -1, :]

    raise ValueError(f"Unsupported logits shape: {tuple(x.shape)}")


def is_all_time_logits_shape(shape):
    """
    Return True if shape looks like all-time logits, i.e. contains class dim and time dim.
    """
    shape = tuple(int(v) for v in shape)

    if len(shape) != 4:
        return False

    if shape == (1, NUM_CLASSES, SEQ_LEN, 1):
        return True

    if shape == (1, SEQ_LEN, 1, NUM_CLASSES):
        return True

    if shape == (1, 1, SEQ_LEN, NUM_CLASSES):
        return True

    return False


# ============================================================
# Model loading
# ============================================================

def build_model_from_checkpoint(ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    if "model_state_dict" not in ckpt:
        raise RuntimeError("Checkpoint missing model_state_dict")

    cfg = ckpt.get("model_config", {})

    print("[info] checkpoint model_config:", cfg)

    model = ECG5000FullQuantTCN(
        num_classes=cfg.get("num_classes", NUM_CLASSES),
        seq_len=cfg.get("seq_len", SEQ_LEN),
        num_inputs=cfg.get("num_inputs", IN_CH),
        num_channels=tuple(cfg.get("num_channels", (16, 16, 32))),
        kernel_size=cfg.get("kernel_size", 3),
        dropout=cfg.get("dropout", 0.0),
        causal=cfg.get("causal", True),
    )

    state = ckpt["model_state_dict"]

    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            clean_state[k[7:]] = v
        else:
            clean_state[k] = v

    result = model.load_state_dict(clean_state, strict=False)

    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    print("[info] Missing keys:", missing)
    print("[info] Unexpected keys:", unexpected)

    serious_missing = [k for k in missing if not k.endswith(".value")]
    serious_unexpected = [k for k in unexpected if not k.endswith(".value")]

    if len(serious_missing) > 0 or len(serious_unexpected) > 0:
        raise RuntimeError(
            "Checkpoint/model mismatch.\n"
            f"Serious missing keys: {serious_missing}\n"
            f"Serious unexpected keys: {serious_unexpected}"
        )

    model.eval()
    return model


# ============================================================
# PyTorch verification
# ============================================================

@torch.no_grad()
def run_pytorch(model, x_np, labels):
    x_t = torch.from_numpy(x_np).to(DEVICE)

    raw_out = model(x_t)
    raw_out = qt_value(raw_out)

    logits = logits_to_last_step_torch(raw_out)

    raw_np = raw_out.detach().cpu().numpy()
    logits_np = logits.detach().cpu().numpy()

    preds = logits_np.argmax(axis=1).astype(np.int32)
    acc = float(np.mean(preds == labels))

    print("\n===================================================")
    print("PyTorch verification")
    print("===================================================")
    print("raw output shape:", raw_np.shape)
    print("last logits shape:", logits_np.shape)
    print("pred distribution:", safe_bincount(preds, minlength=NUM_CLASSES))
    print("accuracy:", acc)
    print("first 10 labels:", labels[:10])
    print("first 10 preds :", preds[:10])
    print("first sample last logits:", logits_np[0])

    if not is_all_time_logits_shape(raw_np.shape):
        print("[warn] PyTorch raw output is not the expected all-time logits layout.")
        print("[warn] Expected one of:")
        print("       (N, 5, 140, 1), (N, 140, 1, 5), (N, 1, 140, 5)")
    else:
        print("[ok] PyTorch raw output keeps all time steps.")

    return raw_np, logits_np, preds


# ============================================================
# ONNX execution
# ============================================================

def pick_graph_output(output_dict, tag):
    """
    Pick logits-like graph output from qonnx execution output dict.
    """
    print(f"\n=== Available outputs: {tag} ===")

    candidates = []

    for name, val in output_dict.items():
        arr = np.asarray(val)
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")

        try:
            _ = logits_to_last_step_np(arr)
            candidates.append((name, arr))
        except Exception:
            pass

    if len(candidates) == 0:
        raise RuntimeError(f"No logits-like output found for {tag}")

    name, arr = candidates[0]
    print(f"[info] Chosen output for {tag}: {name}, shape={arr.shape}")

    return name, arr


def execute_onnx_one(model, x_one):
    input_name = model.graph.input[0].name

    ret = oxe.execute_onnx(
        model,
        {input_name: x_one.astype(np.float32)},
    )

    _, raw = pick_graph_output(ret, tag="single-sample first call")

    return raw


def run_onnx_model(onnx_path: Path, x_np, labels, tag):
    if not onnx_path.exists():
        raise FileNotFoundError(f"{tag} ONNX not found: {onnx_path}")

    print("\n===================================================")
    print(f"{tag} verification")
    print("===================================================")
    print("ONNX:", onnx_path)

    model = ModelWrapper(str(onnx_path))

    input_name = model.graph.input[0].name
    input_shape = model.get_tensor_shape(input_name)

    print("input name :", input_name)
    print("input shape:", input_shape)
    print("input dtype:", model.get_tensor_datatype(input_name))

    print("graph outputs:")
    for o in model.graph.output:
        try:
            print(
                " ",
                o.name,
                model.get_tensor_shape(o.name),
                model.get_tensor_datatype(o.name),
            )
        except Exception:
            print(" ", o.name)

    expected_numel = int(np.prod(input_shape))

    raw_outputs = []
    logits_outputs = []

    for i in range(x_np.shape[0]):
        x_flat = x_np[i].reshape(-1)

        if x_flat.size != expected_numel:
            raise ValueError(
                f"{tag}: input numel mismatch at sample {i}: "
                f"sample has {x_flat.size}, model expects {input_shape}={expected_numel}"
            )

        x_one = x_flat.reshape(input_shape).astype(np.float32)

        ret = oxe.execute_onnx(model, {input_name: x_one})

        if i == 0:
            out_name, raw = pick_graph_output(ret, tag=tag)
        else:
            raw = np.asarray(ret[out_name])

        logits = logits_to_last_step_np(raw)

        raw_outputs.append(np.asarray(raw))
        logits_outputs.append(np.asarray(logits).reshape(1, NUM_CLASSES))

    raw_outputs = np.concatenate(raw_outputs, axis=0)
    logits_outputs = np.concatenate(logits_outputs, axis=0)

    preds = logits_outputs.argmax(axis=1).astype(np.int32)
    acc = float(np.mean(preds == labels))

    print(f"\n[{tag}] raw output shape:", raw_outputs.shape)
    print(f"[{tag}] last logits shape:", logits_outputs.shape)
    print(f"[{tag}] pred distribution:", safe_bincount(preds, minlength=NUM_CLASSES))
    print(f"[{tag}] accuracy:", acc)
    print(f"[{tag}] first 10 preds:", preds[:10])
    print(f"[{tag}] first sample last logits:", logits_outputs[0])

    if not is_all_time_logits_shape(raw_outputs.shape[1:]):
        # raw_outputs has batch concatenated, so per-sample raw shape is raw_outputs.shape[1:]
        print(f"[warn] {tag} raw output may not be all-time logits.")
        print(f"[warn] raw_outputs.shape = {raw_outputs.shape}")
    else:
        print(f"[ok] {tag} raw output keeps all time steps.")

    return raw_outputs, logits_outputs, preds, model


# ============================================================
# Comparison
# ============================================================

def compare_logits(name_a, logits_a, pred_a, name_b, logits_b, pred_b, labels):
    logits_a = np.asarray(logits_a).astype(np.float32)
    logits_b = np.asarray(logits_b).astype(np.float32)

    if logits_a.shape != logits_b.shape:
        print(f"\n[compare] {name_a} vs {name_b}")
        print("shape mismatch:", logits_a.shape, logits_b.shape)
        return

    diff = logits_a - logits_b

    print("\n===================================================")
    print(f"Compare: {name_a} vs {name_b}")
    print("===================================================")
    print("logits shape:", logits_a.shape)
    print("max abs diff :", float(np.max(np.abs(diff))))
    print("mean abs diff:", float(np.mean(np.abs(diff))))
    print("allclose     :", bool(np.allclose(logits_a, logits_b, atol=ATOL, rtol=RTOL)))
    print("pred match   :", float(np.mean(pred_a == pred_b)))
    print(f"{name_a} acc:", float(np.mean(pred_a == labels)))
    print(f"{name_b} acc:", float(np.mean(pred_b == labels)))

    print("\nfirst 10 compare:")
    for i in range(min(10, len(labels))):
        print(
            f"i={i} label={int(labels[i])} "
            f"{name_a}_pred={int(pred_a[i])} {name_b}_pred={int(pred_b[i])}"
        )
        print(f"  {name_a} logits:", logits_a[i])
        print(f"  {name_b} logits:", logits_b[i])
        print("  diff:", diff[i])


# ============================================================
# MVAU inspection
# ============================================================

def inspect_mvau_classifier(onnx_path: Path, tag):
    if not onnx_path.exists():
        print(f"[warn] Cannot inspect MVAU. File not found: {onnx_path}")
        return

    print("\n===================================================")
    print(f"MVAU classifier inspection: {tag}")
    print("===================================================")

    model = ModelWrapper(str(onnx_path))

    found = False

    for n in model.graph.node:
        if "MVAU" not in n.op_type and "MatrixVector" not in n.op_type:
            continue

        try:
            cop = getCustomOp(n)
        except Exception:
            continue

        attrs = {}

        for a in [
            "MW",
            "MH",
            "SIMD",
            "PE",
            "inputDataType",
            "weightDataType",
            "accDataType",
            "outputDataType",
            "numInputVectors",
            "noActivation",
            "ActVal",
        ]:
            try:
                attrs[a] = cop.get_nodeattr(a)
            except Exception:
                pass

        mh = attrs.get("MH", None)

        if mh == NUM_CLASSES:
            found = True

            print("\nnode:", n.name if n.name else "<unnamed>")
            print("op_type:", n.op_type)
            print("domain:", n.domain)
            print("inputs:", list(n.input))
            print("outputs:", list(n.output))
            print("attrs:", attrs)

            mw = attrs.get("MW", None)
            niv = attrs.get("numInputVectors", None)

            if mw == 32:
                print("[ok] Classifier MW is 32, expected for pointwise classifier.")
            elif mw == 4480:
                print("[bad] Classifier MW is 4480. This is old flatten classifier behavior.")
            else:
                print(f"[info] Classifier MW is {mw}.")

            if niv is not None:
                print("[info] numInputVectors:", niv)

            for t in list(n.input) + list(n.output):
                try:
                    print(
                        "  tensor",
                        t,
                        "shape",
                        model.get_tensor_shape(t),
                        "dtype",
                        model.get_tensor_datatype(t),
                    )
                except Exception:
                    pass

    if not found:
        print("No MVAU/MatrixVector node with MH=5 found.")
        print("This can be normal before step_convert_to_hw.")


# ============================================================
# Save verification outputs
# ============================================================

def save_outputs(indices, labels, x_np, pyt_raw, pyt_logits, pyt_preds,
                 clean_raw=None, clean_logits=None, clean_preds=None,
                 finn_raw=None, finn_logits=None, finn_preds=None):
    np.save(str(OUT_DIR / "selected_indices.npy"), indices.astype(np.int32))
    np.save(str(OUT_DIR / "labels.npy"), labels.astype(np.int32))
    np.save(str(OUT_DIR / "x_selected.npy"), x_np.astype(np.float32))

    np.save(str(OUT_DIR / "pytorch_raw_all_time_logits.npy"), pyt_raw.astype(np.float32))
    np.save(str(OUT_DIR / "pytorch_last_logits.npy"), pyt_logits.astype(np.float32))
    np.save(str(OUT_DIR / "pytorch_preds.npy"), pyt_preds.astype(np.int32))

    if clean_raw is not None:
        np.save(str(OUT_DIR / "clean_qonnx_raw_all_time_logits.npy"), clean_raw.astype(np.float32))
        np.save(str(OUT_DIR / "clean_qonnx_last_logits.npy"), clean_logits.astype(np.float32))
        np.save(str(OUT_DIR / "clean_qonnx_preds.npy"), clean_preds.astype(np.int32))

    if finn_raw is not None:
        np.save(str(OUT_DIR / "finn_streamlined_raw_all_time_logits.npy"), finn_raw.astype(np.float32))
        np.save(str(OUT_DIR / "finn_streamlined_last_logits.npy"), finn_logits.astype(np.float32))
        np.save(str(OUT_DIR / "finn_streamlined_preds.npy"), finn_preds.astype(np.int32))

    csv_cols = [
        indices.astype(np.int32),
        labels.astype(np.int32),
        pyt_preds.astype(np.int32),
    ]

    header = ["index", "label", "pytorch_pred"]

    if clean_preds is not None:
        csv_cols.append(clean_preds.astype(np.int32))
        header.append("clean_qonnx_pred")

    if finn_preds is not None:
        csv_cols.append(finn_preds.astype(np.int32))
        header.append("finn_streamlined_pred")

    csv = np.stack(csv_cols, axis=1)

    np.savetxt(
        str(OUT_DIR / "verify_summary.csv"),
        csv,
        fmt="%d",
        delimiter=",",
        header=",".join(header),
        comments="",
    )

    print("\n[info] Saved verification outputs to:")
    print(" ", OUT_DIR)


# ============================================================
# Main
# ============================================================

def main():
    print("===================================================")
    print("Verify pointwise QTCN export")
    print("===================================================")
    print("PROJECT_DIR:", PROJECT_DIR)
    print("CHECKPOINT :", CHECKPOINT_PATH)
    print("TEST_FILE  :", TEST_FILE)
    print("CLEAN_QONNX:", CLEAN_QONNX_PATH)
    print("FINN_ONNX  :", FINN_ONNX_STREAMLINED_PATH)
    print("OUT_DIR    :", OUT_DIR)

    torch.manual_seed(0)
    np.random.seed(0)

    model = build_model_from_checkpoint(CHECKPOINT_PATH).to(DEVICE)
    model.eval()

    x_all, labels_all = load_ecg5000_txt(TEST_FILE)
    x_np, labels, indices = select_subset(x_all, labels_all)

    print("\n[info] Selected input shape:", x_np.shape)
    print("[info] Selected labels shape:", labels.shape)
    print("[info] First sample first 10 raw values:")
    print(x_np[0].reshape(-1)[:10])

    pyt_raw, pyt_logits, pyt_preds = run_pytorch(model, x_np, labels)

    clean_raw = None
    clean_logits = None
    clean_preds = None
    clean_model = None

    if CLEAN_QONNX_PATH.exists():
        clean_raw, clean_logits, clean_preds, clean_model = run_onnx_model(
            CLEAN_QONNX_PATH,
            x_np,
            labels,
            tag="CLEAN_QONNX",
        )

        compare_logits(
            "PyTorch",
            pyt_logits,
            pyt_preds,
            "CLEAN_QONNX",
            clean_logits,
            clean_preds,
            labels,
        )
    else:
        print("[warn] CLEAN_QONNX not found, skipped.")

    finn_raw = None
    finn_logits = None
    finn_preds = None
    finn_model = None

    if FINN_ONNX_STREAMLINED_PATH.exists():
        finn_raw, finn_logits, finn_preds, finn_model = run_onnx_model(
            FINN_ONNX_STREAMLINED_PATH,
            x_np,
            labels,
            tag="FINN_STREAMLINED",
        )

        if clean_logits is not None:
            compare_logits(
                "CLEAN_QONNX",
                clean_logits,
                clean_preds,
                "FINN_STREAMLINED",
                finn_logits,
                finn_preds,
                labels,
            )

        compare_logits(
            "PyTorch",
            pyt_logits,
            pyt_preds,
            "FINN_STREAMLINED",
            finn_logits,
            finn_preds,
            labels,
        )

    else:
        print("[warn] FINN_ONNX_STREAMLINED not found, skipped.")

    inspect_mvau_classifier(CLEAN_QONNX_PATH, "CLEAN_QONNX")
    inspect_mvau_classifier(FINN_ONNX_STREAMLINED_PATH, "FINN_STREAMLINED")

    save_outputs(
        indices=indices,
        labels=labels,
        x_np=x_np,
        pyt_raw=pyt_raw,
        pyt_logits=pyt_logits,
        pyt_preds=pyt_preds,
        clean_raw=clean_raw,
        clean_logits=clean_logits,
        clean_preds=clean_preds,
        finn_raw=finn_raw,
        finn_logits=finn_logits,
        finn_preds=finn_preds,
    )

    print("\n===================================================")
    print("Final checklist")
    print("===================================================")

    print("1. PyTorch raw output shape:")
    print("   ", pyt_raw.shape)

    if is_all_time_logits_shape(pyt_raw.shape):
        print("   [ok] PyTorch output keeps all time steps.")
    else:
        print("   [warn] PyTorch output is not expected all-time layout.")

    if clean_raw is not None:
        print("2. CLEAN_QONNX raw output shape:")
        print("   ", clean_raw.shape)
        if is_all_time_logits_shape(clean_raw.shape[1:]):
            print("   [ok] CLEAN_QONNX output keeps all time steps.")
        else:
            print("   [warn] CLEAN_QONNX output may not be all-time layout.")

    if finn_raw is not None:
        print("3. FINN_STREAMLINED raw output shape:")
        print("   ", finn_raw.shape)
        if is_all_time_logits_shape(finn_raw.shape[1:]):
            print("   [ok] FINN_STREAMLINED output keeps all time steps.")
        else:
            print("   [warn] FINN_STREAMLINED output may not be all-time layout.")

    print("\nExpected for new model:")
    print("  raw output should keep all time steps:")
    print("    [N, 5, 140, 1] or equivalent layout")
    print("  last-step logits used for classification:")
    print("    logits[:, :, -1, 0]")
    print("  after FINN step_convert_to_hw, final classifier should be:")
    print("    MW=32, MH=5, numInputVectors includes 140")
    print("  it should NOT be:")
    print("    MW=4480, MH=5")


if __name__ == "__main__":
    main()