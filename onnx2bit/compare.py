import os
import sys
import csv
from pathlib import Path
from collections import Counter

import numpy as np
import torch

from qonnx.core.modelwrapper import ModelWrapper
import qonnx.core.onnx_exec as oxe


# ============================================================
# Paths
# ============================================================

# Assumption:
# this script is placed under:
#   finn/notebooks/onnx2bit/compare_pytorch_qonnx_fpga_results.py
THIS_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = THIS_DIR.parent
PROJECT_DIR = NOTEBOOKS_DIR / "pytorch-tcn"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "new_version"))

# Use the same model import as your export script.
from new_version.newnet import ECG5000FullQuantTCN


# ============================================================
# User config
# ============================================================

NUM_CLASSES = 5
SEQ_LEN = 140
NUM_INPUTS = 1
NUM_CHANNELS = (16, 16, 32)
KERNEL_SIZE = 3
DROPOUT = 0.1
CAUSAL = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the checkpoint that was used for QONNX export / bitfile build.
CHECKPOINT_PATH = PROJECT_DIR / "new_version" / "best_qtcn_model_last_step.pth"

# Use the QONNX file that you want to compare against PyTorch.
# For PyTorch vs QONNX, usually use the cleaned shaped fixed QONNX.
QONNX_PATH = PROJECT_DIR / "onnx_exports" / "ecg5000_pointwise_qtcn_qonnx_clean_shaped_fixed.onnx"

# ECG5000 test file.
TEST_TXT = PROJECT_DIR / "datasets" / "ECG5000" / "ECG5000_TEST.txt"

# Copy this folder from PYNQ if running this script on your PC/FINN Docker.
# Example on PYNQ:
#   /home/xilinx/jupyter_notebooks/Thesis/results_pointwise
# Copy to local:
#   /home/slowman/Desktop/project/Thesis/finn/notebooks/pytorch-tcn/results_pointwise
FPGA_RESULTS_DIR = PROJECT_DIR / "results_pointwise"

FPGA_LAST_RAW_NPY = FPGA_RESULTS_DIR / "fpga_last_step_logits_raw.npy"
FPGA_LAST_POST_NPY = FPGA_RESULTS_DIR / "fpga_last_step_logits_post.npy"
FPGA_PREDS_NPY = FPGA_RESULTS_DIR / "fpga_preds.npy"
FPGA_LABELS_NPY = FPGA_RESULTS_DIR / "ecg5000_test_labels.npy"
FPGA_CSV = FPGA_RESULTS_DIR / "fpga_ecg5000_pointwise_result.csv"

OUT_COMPARE_CSV = FPGA_RESULTS_DIR / "compare_pytorch_qonnx_fpga.csv"
OUT_SUMMARY_TXT = FPGA_RESULTS_DIR / "compare_pytorch_qonnx_fpga_summary.txt"

# If None, compare all FPGA samples available.
# For quick debug, set e.g. MAX_COMPARE_SAMPLES = 20.
MAX_COMPARE_SAMPLES = None

# The FPGA raw INT24 output usually needs affine correction to match PyTorch logits.
# If fpga_last_step_logits_post.npy exists, this script uses it directly.
APPLY_OUTPUT_AFFINE_TO_FPGA_RAW_IF_NEEDED = False
OUTPUT_A = np.array(
    [7.344037e-05, 7.337188e-05, 8.323102e-05, 8.668597e-05, 7.477590e-05],
    dtype=np.float32,
)
OUTPUT_B = np.array(
    [0.7575136, -0.680948, -0.0545594, -0.50195503, 0.00481556],
    dtype=np.float32,
)

# Set this to True only if you know the QONNX output is also raw INT accumulator
# and needs the same affine correction. Usually keep False.
APPLY_OUTPUT_AFFINE_TO_QONNX = False
APPLY_OUTPUT_AFFINE_TO_PYTORCH = False


# ============================================================
# Helpers
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
        raise FileNotFoundError(f"TEST_TXT not found: {path}")

    try:
        data = np.loadtxt(str(path), dtype=np.float32)
    except Exception:
        data = np.loadtxt(str(path), dtype=np.float32, delimiter=",")

    if data.ndim != 2 or data.shape[1] != 141:
        raise ValueError(f"Expected ECG5000 txt shape [N, 141], got {data.shape}")

    y_raw = data[:, 0].astype(np.int32)
    x_raw = data[:, 1:].astype(np.float32)

    unique = sorted(np.unique(y_raw).astype(np.int32).tolist())
    if unique == [1, 2, 3, 4, 5]:
        y = y_raw - 1
    elif unique == [0, 1, 2, 3, 4]:
        y = y_raw
    else:
        label_map = {int(v): i for i, v in enumerate(unique)}
        y = np.asarray([label_map[int(v)] for v in y_raw], dtype=np.int32)
        print("[warn] Non-standard labels. label_map =", label_map)

    return x_raw.astype(np.float32), y.astype(np.int32)


def read_fpga_indices_from_csv(csv_path: Path):
    if not csv_path.exists():
        return None

    indices = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        if "index" not in reader.fieldnames:
            return None
        for row in reader:
            indices.append(int(row["index"]))

    return np.asarray(indices, dtype=np.int32)


def affine_output(x):
    x = np.asarray(x, dtype=np.float32)
    return x * OUTPUT_A.reshape(1, NUM_CLASSES) + OUTPUT_B.reshape(1, NUM_CLASSES)


def logits_to_2d_torch(x):
    x = qt_value(x)

    if x.dim() == 2:
        if x.shape[1] != NUM_CLASSES:
            raise ValueError(f"Expected [N, {NUM_CLASSES}], got {tuple(x.shape)}")
        return x

    if x.dim() == 4:
        # Desired output: [N, C, 1, 1]
        if x.shape[1] == NUM_CLASSES and x.shape[2] == 1 and x.shape[3] == 1:
            return x[:, :, 0, 0]

        # Old pointwise output: [N, C, L, 1], use last time step.
        if x.shape[1] == NUM_CLASSES and x.shape[3] == 1:
            return x[:, :, -1, 0]

        # NHWC-like: [N, L, 1, C]
        if x.shape[-1] == NUM_CLASSES and x.shape[2] == 1:
            return x[:, -1, 0, :]

        # [N, 1, L, C]
        if x.shape[-1] == NUM_CLASSES and x.shape[1] == 1:
            return x[:, 0, -1, :]

    raise ValueError(f"Cannot convert torch logits with shape {tuple(x.shape)} to [N, C]")


def logits_to_2d_np(x):
    x = np.asarray(x)

    if x.ndim == 1 and x.size == NUM_CLASSES:
        return x.reshape(1, NUM_CLASSES)

    if x.ndim == 2:
        if x.shape[1] != NUM_CLASSES:
            raise ValueError(f"Expected [N, {NUM_CLASSES}], got {x.shape}")
        return x

    if x.ndim == 4:
        # Desired NCHW output: [N, C, 1, 1]
        if x.shape[1] == NUM_CLASSES and x.shape[2] == 1 and x.shape[3] == 1:
            return x[:, :, 0, 0]

        # Old NCHW pointwise output: [N, C, L, 1]
        if x.shape[1] == NUM_CLASSES and x.shape[3] == 1:
            return x[:, :, -1, 0]

        # FINN/PYNQ NHWC-like output: [N, L, 1, C]
        if x.shape[-1] == NUM_CLASSES and x.shape[2] == 1:
            return x[:, -1, 0, :]

        # Possible layout: [N, 1, L, C]
        if x.shape[-1] == NUM_CLASSES and x.shape[1] == 1:
            return x[:, 0, -1, :]

    raise ValueError(f"Cannot convert numpy logits with shape {x.shape} to [N, C]")


def pick_logits_output_from_onnx_outputs(output_dict, tag="QONNX"):
    print(f"\n=== Available ONNX outputs: {tag} ===")
    for name, val in output_dict.items():
        print(f"{name}: shape={np.asarray(val).shape}")

    candidates = []
    for name, val in output_dict.items():
        arr = np.asarray(val)
        try:
            logits = logits_to_2d_np(arr)
            if logits.shape[1] == NUM_CLASSES:
                candidates.append((name, arr, logits))
        except Exception:
            pass

    if len(candidates) == 0:
        raise RuntimeError("No logits-like QONNX output found. See printed shapes above.")

    name, raw, logits = candidates[0]
    print(f"Chosen output: {name}, raw shape={raw.shape}, logits shape={logits.shape}")
    return raw, logits


def build_model_from_checkpoint(ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg = ckpt.get("model_config", {})

    print("[info] Checkpoint:", ckpt_path)
    print("[info] checkpoint model_config:", cfg)

    model = ECG5000FullQuantTCN(
        num_classes=cfg.get("num_classes", NUM_CLASSES),
        seq_len=cfg.get("seq_len", SEQ_LEN),
        num_inputs=cfg.get("num_inputs", NUM_INPUTS),
        num_channels=tuple(cfg.get("num_channels", NUM_CHANNELS)),
        kernel_size=cfg.get("kernel_size", KERNEL_SIZE),
        dropout=cfg.get("dropout", DROPOUT),
        causal=cfg.get("causal", CAUSAL),
    )

    sd = ckpt["model_state_dict"]
    clean_sd = {}
    for k, v in sd.items():
        clean_sd[k[7:] if k.startswith("module.") else k] = v

    load_res = model.load_state_dict(clean_sd, strict=False)
    missing = list(load_res.missing_keys)
    unexpected = list(load_res.unexpected_keys)

    print("[info] Missing keys   :", missing)
    print("[info] Unexpected keys:", unexpected)

    return model


@torch.no_grad()
def run_pytorch(model, x_raw, batch_size=128):
    model.eval()
    model.to(DEVICE)

    outputs = []
    n = x_raw.shape[0]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = torch.tensor(x_raw[start:end], dtype=torch.float32).unsqueeze(1).unsqueeze(-1).to(DEVICE)
        out = model(xb)
        logits = logits_to_2d_torch(out).detach().cpu().numpy().astype(np.float32)
        outputs.append(logits)

    logits = np.concatenate(outputs, axis=0)

    if APPLY_OUTPUT_AFFINE_TO_PYTORCH:
        logits = affine_output(logits)

    preds = logits.argmax(axis=1).astype(np.int32)
    return logits, preds


def run_qonnx(qonnx_path: Path, x_raw):
    if not qonnx_path.exists():
        raise FileNotFoundError(f"QONNX not found: {qonnx_path}")

    model = ModelWrapper(str(qonnx_path))
    input_name = model.graph.input[0].name

    logits_list = []
    raw_shape_first = None

    for i in range(x_raw.shape[0]):
        xb = x_raw[i:i + 1].astype(np.float32).reshape(1, 1, SEQ_LEN, 1)
        out_dict = oxe.execute_onnx(model, {input_name: xb})

        if i == 0:
            raw, logits = pick_logits_output_from_onnx_outputs(out_dict, tag="QONNX first sample")
            raw_shape_first = np.asarray(raw).shape
        else:
            # Use the first logits-like output for speed / consistency.
            logits = None
            for _, val in out_dict.items():
                try:
                    logits = logits_to_2d_np(val)
                    break
                except Exception:
                    continue
            if logits is None:
                raise RuntimeError(f"No logits-like output found for QONNX sample {i}")

        logits_list.append(np.asarray(logits, dtype=np.float32).reshape(1, NUM_CLASSES))

        if (i + 1) % 100 == 0 or (i + 1) == x_raw.shape[0]:
            print(f"[qonnx] executed {i + 1}/{x_raw.shape[0]} samples")

    logits = np.concatenate(logits_list, axis=0)

    if APPLY_OUTPUT_AFFINE_TO_QONNX:
        logits = affine_output(logits)

    preds = logits.argmax(axis=1).astype(np.int32)
    print("[info] First QONNX raw output shape:", raw_shape_first)
    return logits, preds


def load_fpga_results():
    if not FPGA_RESULTS_DIR.exists():
        raise FileNotFoundError(f"FPGA_RESULTS_DIR not found: {FPGA_RESULTS_DIR}")

    if not FPGA_LABELS_NPY.exists():
        raise FileNotFoundError(f"FPGA labels not found: {FPGA_LABELS_NPY}")
    if not FPGA_PREDS_NPY.exists():
        raise FileNotFoundError(f"FPGA preds not found: {FPGA_PREDS_NPY}")

    labels = np.load(str(FPGA_LABELS_NPY)).astype(np.int32).reshape(-1)
    preds = np.load(str(FPGA_PREDS_NPY)).astype(np.int32).reshape(-1)

    fpga_raw = None
    fpga_post = None

    if FPGA_LAST_RAW_NPY.exists():
        fpga_raw = logits_to_2d_np(np.load(str(FPGA_LAST_RAW_NPY))).astype(np.float32)

    if FPGA_LAST_POST_NPY.exists():
        fpga_post = logits_to_2d_np(np.load(str(FPGA_LAST_POST_NPY))).astype(np.float32)

    if fpga_post is not None:
        logits_for_compare = fpga_post
        logits_source = "fpga_last_step_logits_post.npy"
    elif fpga_raw is not None:
        if APPLY_OUTPUT_AFFINE_TO_FPGA_RAW_IF_NEEDED:
            logits_for_compare = affine_output(fpga_raw)
            logits_source = "fpga_raw_with_affine"
        else:
            logits_for_compare = fpga_raw
            logits_source = "fpga_last_step_logits_raw.npy"
    else:
        raise FileNotFoundError("Neither FPGA raw nor post logits npy found.")

    indices = read_fpga_indices_from_csv(FPGA_CSV)
    if indices is None:
        indices = np.arange(len(labels), dtype=np.int32)
        print("[warn] FPGA CSV with indices not found. Assuming first N test samples.")

    return {
        "indices": indices,
        "labels": labels,
        "preds": preds,
        "raw_logits": fpga_raw,
        "post_logits": fpga_post,
        "logits_for_compare": logits_for_compare,
        "logits_source": logits_source,
    }


def accuracy(preds, labels):
    return float(np.mean(np.asarray(preds).reshape(-1) == np.asarray(labels).reshape(-1)))


def agreement(a, b):
    return float(np.mean(np.asarray(a).reshape(-1) == np.asarray(b).reshape(-1)))


def diff_stats(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    d = np.abs(a - b)
    return {
        "max_abs": float(np.max(d)),
        "mean_abs": float(np.mean(d)),
        "median_abs": float(np.median(d)),
        "p95_abs": float(np.percentile(d, 95)),
    }


def per_class_report(name, preds, labels):
    lines = []
    lines.append(f"\n{name} per-class accuracy:")
    for c in range(NUM_CLASSES):
        mask = labels == c
        total = int(np.sum(mask))
        corr = int(np.sum(preds[mask] == labels[mask])) if total > 0 else 0
        acc = corr / total if total > 0 else 0.0
        lines.append(f"  class {c}: {corr}/{total}, acc={acc:.4f}")
    return lines


def save_comparison_csv(indices, labels, pyt_preds, qonnx_preds, fpga_preds,
                        pyt_logits, qonnx_logits, fpga_logits):
    rows = []
    for i in range(len(labels)):
        row = {
            "sample_index": int(indices[i]),
            "label": int(labels[i]),
            "pytorch_pred": int(pyt_preds[i]),
            "qonnx_pred": int(qonnx_preds[i]),
            "fpga_pred": int(fpga_preds[i]),
            "pytorch_correct": int(pyt_preds[i] == labels[i]),
            "qonnx_correct": int(qonnx_preds[i] == labels[i]),
            "fpga_correct": int(fpga_preds[i] == labels[i]),
            "pytorch_qonnx_match": int(pyt_preds[i] == qonnx_preds[i]),
            "pytorch_fpga_match": int(pyt_preds[i] == fpga_preds[i]),
            "qonnx_fpga_match": int(qonnx_preds[i] == fpga_preds[i]),
        }

        for c in range(NUM_CLASSES):
            row[f"pytorch_logit_{c}"] = float(pyt_logits[i, c])
            row[f"qonnx_logit_{c}"] = float(qonnx_logits[i, c])
            row[f"fpga_logit_{c}"] = float(fpga_logits[i, c])

        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(OUT_COMPARE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[info] Saved comparison CSV:", OUT_COMPARE_CSV)


def main():
    print("===================================================")
    print("Compare PyTorch / QONNX / FPGA bit results")
    print("===================================================")
    print("THIS_DIR        :", THIS_DIR)
    print("PROJECT_DIR     :", PROJECT_DIR)
    print("CHECKPOINT_PATH :", CHECKPOINT_PATH)
    print("QONNX_PATH      :", QONNX_PATH)
    print("TEST_TXT        :", TEST_TXT)
    print("FPGA_RESULTS_DIR:", FPGA_RESULTS_DIR)
    print("DEVICE          :", DEVICE)

    fpga = load_fpga_results()

    indices = fpga["indices"]
    labels_fpga = fpga["labels"]
    preds_fpga = fpga["preds"]
    logits_fpga = fpga["logits_for_compare"]

    n = len(labels_fpga)
    if MAX_COMPARE_SAMPLES is not None:
        n = min(n, int(MAX_COMPARE_SAMPLES))
        indices = indices[:n]
        labels_fpga = labels_fpga[:n]
        preds_fpga = preds_fpga[:n]
        logits_fpga = logits_fpga[:n]

    x_all, y_all = load_ecg5000_txt(TEST_TXT)

    if np.max(indices) >= len(y_all):
        raise ValueError("FPGA sample indices exceed TEST_TXT length.")

    x_sel = x_all[indices]
    labels = y_all[indices]

    if not np.array_equal(labels, labels_fpga):
        print("[warn] Labels from TEST_TXT do not match FPGA labels npy.")
        print("       TEST labels distribution:", safe_bincount(labels, NUM_CLASSES))
        print("       FPGA labels distribution:", safe_bincount(labels_fpga, NUM_CLASSES))
        print("       Using FPGA labels for accuracy to match board result.")
        labels = labels_fpga

    print("\n[info] Number of compared samples:", n)
    print("[info] indices first 20:", indices[:20])
    print("[info] label distribution:", safe_bincount(labels, NUM_CLASSES))
    print("[info] FPGA logits source:", fpga["logits_source"])

    model = build_model_from_checkpoint(CHECKPOINT_PATH)

    print("\n=== Running PyTorch ===")
    logits_pyt, preds_pyt = run_pytorch(model, x_sel)
    print("PyTorch logits shape:", logits_pyt.shape)

    print("\n=== Running QONNX ===")
    logits_qonnx, preds_qonnx = run_qonnx(QONNX_PATH, x_sel)
    print("QONNX logits shape:", logits_qonnx.shape)

    print("\n=== FPGA loaded results ===")
    print("FPGA logits shape:", logits_fpga.shape)
    print("FPGA preds shape :", preds_fpga.shape)

    # Make sure all lengths match.
    min_n = min(len(labels), len(preds_pyt), len(preds_qonnx), len(preds_fpga),
                logits_pyt.shape[0], logits_qonnx.shape[0], logits_fpga.shape[0])
    if min_n != len(labels):
        print(f"[warn] Length mismatch. Trimming all arrays to {min_n}.")
        indices = indices[:min_n]
        labels = labels[:min_n]
        preds_pyt = preds_pyt[:min_n]
        preds_qonnx = preds_qonnx[:min_n]
        preds_fpga = preds_fpga[:min_n]
        logits_pyt = logits_pyt[:min_n]
        logits_qonnx = logits_qonnx[:min_n]
        logits_fpga = logits_fpga[:min_n]

    lines = []
    lines.append("===================================================")
    lines.append("Summary")
    lines.append("===================================================")
    lines.append(f"Compared samples: {len(labels)}")
    lines.append(f"Label distribution: {safe_bincount(labels, NUM_CLASSES).tolist()}")
    lines.append("")

    lines.append("Accuracy:")
    lines.append(f"  PyTorch: {accuracy(preds_pyt, labels):.6f}")
    lines.append(f"  QONNX  : {accuracy(preds_qonnx, labels):.6f}")
    lines.append(f"  FPGA   : {accuracy(preds_fpga, labels):.6f}")
    lines.append("")

    lines.append("Prediction agreement:")
    lines.append(f"  PyTorch vs QONNX: {agreement(preds_pyt, preds_qonnx):.6f}")
    lines.append(f"  PyTorch vs FPGA : {agreement(preds_pyt, preds_fpga):.6f}")
    lines.append(f"  QONNX   vs FPGA : {agreement(preds_qonnx, preds_fpga):.6f}")
    lines.append("")

    lines.append("Logit difference stats:")
    for name, a, b in [
        ("PyTorch vs QONNX", logits_pyt, logits_qonnx),
        ("PyTorch vs FPGA ", logits_pyt, logits_fpga),
        ("QONNX   vs FPGA ", logits_qonnx, logits_fpga),
    ]:
        st = diff_stats(a, b)
        lines.append(
            f"  {name}: max={st['max_abs']:.6g}, mean={st['mean_abs']:.6g}, "
            f"median={st['median_abs']:.6g}, p95={st['p95_abs']:.6g}"
        )
    lines.append("")

    lines += per_class_report("PyTorch", preds_pyt, labels)
    lines += per_class_report("QONNX", preds_qonnx, labels)
    lines += per_class_report("FPGA", preds_fpga, labels)

    lines.append("\nPrediction distributions:")
    lines.append(f"  PyTorch: {safe_bincount(preds_pyt, NUM_CLASSES).tolist()}")
    lines.append(f"  QONNX  : {safe_bincount(preds_qonnx, NUM_CLASSES).tolist()}")
    lines.append(f"  FPGA   : {safe_bincount(preds_fpga, NUM_CLASSES).tolist()}")

    summary = "\n".join(lines)
    print("\n" + summary)

    FPGA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_SUMMARY_TXT, "w") as f:
        f.write(summary + "\n")
    print("[info] Saved summary:", OUT_SUMMARY_TXT)

    save_comparison_csv(
        indices=indices,
        labels=labels,
        pyt_preds=preds_pyt,
        qonnx_preds=preds_qonnx,
        fpga_preds=preds_fpga,
        pyt_logits=logits_pyt,
        qonnx_logits=logits_qonnx,
        fpga_logits=logits_fpga,
    )

    # Print first few detailed rows for quick visual inspection.
    print("\n=== First 10 samples detail ===")
    for i in range(min(10, len(labels))):
        print(
            f"idx={int(indices[i])} label={int(labels[i])} "
            f"pyt={int(preds_pyt[i])} qonnx={int(preds_qonnx[i])} fpga={int(preds_fpga[i])}"
        )
        print("  pyt  :", logits_pyt[i])
        print("  qonnx:", logits_qonnx[i])
        print("  fpga :", logits_fpga[i])


if __name__ == "__main__":
    main()
