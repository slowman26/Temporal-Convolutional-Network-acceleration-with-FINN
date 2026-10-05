import sys
import csv
import json
from pathlib import Path

import numpy as np
import torch

from qonnx.core.modelwrapper import ModelWrapper
import qonnx.core.onnx_exec as oxe


# ============================================================
# Paths
# ============================================================

# Assumption:
# this script is placed under:
#   finn/notebooks/onnx2bit/fit_fpga_output_affine_correction.py
THIS_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = THIS_DIR.parent
PROJECT_DIR = NOTEBOOKS_DIR / "pytorch-tcn"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "new_version"))

# Optional PyTorch target. QONNX is the default target.
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

# Recommended:
#   1. Run the PYNQ board test on validation/calibration samples.
#   2. Copy the board result folder to FPGA_RESULTS_DIR.
#   3. Fit A/B using QONNX or PyTorch target on the same sample indices.
#
# For quick debug, you can use TEST_TXT and the board test results on test set,
# but for thesis final numbers, fit on validation/calibration data instead.
CALIB_TXT = PROJECT_DIR / "datasets" / "ECG5000" / "ECG5000_val_split.txt"
# If your board result was generated from ECG5000_TEST.txt, use this instead:
# CALIB_TXT = PROJECT_DIR / "datasets" / "ECG5000" / "ECG5000_TEST.txt"

# QONNX model used as floating target.
# Use the exact QONNX that corresponds to the bitfile.
QONNX_PATH = PROJECT_DIR / "onnx_exports" / "ecg5000_pointwise_qtcn_qonnx_clean_shaped_fixed.onnx"

# PyTorch checkpoint is optional. Used only if TARGET_SOURCE = "pytorch".
CHECKPOINT_PATH = PROJECT_DIR / "new_version" / "best_qtcn_model.pth"

# Copied board results folder.
# It must contain at least:
#   fpga_last_step_logits_raw.npy
#   ecg5000_test_labels.npy or equivalent labels file
# Optional but recommended:
#   fpga_ecg5000_pointwise_result.csv with sample index column
FPGA_RESULTS_DIR = PROJECT_DIR / "results_pointwise"

FPGA_RAW_NPY = FPGA_RESULTS_DIR / "fpga_last_step_logits_raw.npy"
FPGA_PREDS_NPY = FPGA_RESULTS_DIR / "fpga_preds.npy"
FPGA_LABELS_NPY = FPGA_RESULTS_DIR / "ecg5000_test_labels.npy"
FPGA_CSV = FPGA_RESULTS_DIR / "fpga_ecg5000_pointwise_result.csv"

OUT_JSON = FPGA_RESULTS_DIR / "fitted_output_affine.json"
OUT_NPY_A = FPGA_RESULTS_DIR / "fitted_OUTPUT_A.npy"
OUT_NPY_B = FPGA_RESULTS_DIR / "fitted_OUTPUT_B.npy"
OUT_TXT = FPGA_RESULTS_DIR / "fitted_output_affine_snippet.txt"
OUT_COMPARE_CSV = FPGA_RESULTS_DIR / "fitted_affine_compare.csv"

# Choose target source for fitting:
#   "qonnx"   : recommended, target is exported QONNX floating logits
#   "pytorch" : target is PyTorch model floating logits
TARGET_SOURCE = "qonnx"

# Fit/evaluation split.
# "all"       : fit on all available samples and report on all samples.
# "holdout"   : fit on first FIT_RATIO samples and evaluate on the rest.
#               useful to check whether correction generalizes.
FIT_MODE = "all"
FIT_RATIO = 0.7

# If your FPGA result was produced on only the first N samples and no csv exists,
# this script assumes indices = [0, 1, ..., N-1].
MAX_SAMPLES = None

# Numerical safety.
MIN_DENOM = 1e-12


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


def accuracy(preds, labels):
    preds = np.asarray(preds).astype(np.int32).reshape(-1)
    labels = np.asarray(labels).astype(np.int32).reshape(-1)
    return float(np.mean(preds == labels))


def agreement(a, b):
    a = np.asarray(a).astype(np.int32).reshape(-1)
    b = np.asarray(b).astype(np.int32).reshape(-1)
    return float(np.mean(a == b))


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


def load_ecg_txt(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"CALIB_TXT not found: {path}")

    try:
        data = np.loadtxt(str(path), dtype=np.float32)
    except Exception:
        data = np.loadtxt(str(path), dtype=np.float32, delimiter=",")

    if data.ndim != 2 or data.shape[1] != 141:
        raise ValueError(f"Expected ECG5000 txt shape [N, 141], got {data.shape}")

    labels_raw = data[:, 0].astype(np.int32)
    x_raw = data[:, 1:].astype(np.float32)

    unique = sorted(np.unique(labels_raw).astype(np.int32).tolist())
    if unique == [1, 2, 3, 4, 5]:
        labels = labels_raw - 1
    elif unique == [0, 1, 2, 3, 4]:
        labels = labels_raw
    else:
        label_map = {int(v): i for i, v in enumerate(unique)}
        labels = np.asarray([label_map[int(v)] for v in labels_raw], dtype=np.int32)
        print("[warn] Non-standard labels detected. label_map =", label_map)

    return x_raw.astype(np.float32), labels.astype(np.int32)


def read_fpga_indices_from_csv(csv_path: Path):
    if not csv_path.exists():
        return None

    indices = []
    labels = []
    preds = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "index" not in fieldnames:
            return None

        for row in reader:
            indices.append(int(row["index"]))
            if "label" in row:
                labels.append(int(row["label"]))
            if "pred" in row:
                preds.append(int(row["pred"]))

    out = {"indices": np.asarray(indices, dtype=np.int32)}
    if len(labels) == len(indices):
        out["labels"] = np.asarray(labels, dtype=np.int32)
    if len(preds) == len(indices):
        out["preds"] = np.asarray(preds, dtype=np.int32)

    return out


def logits_to_2d_np(x):
    x = np.asarray(x)

    if x.ndim == 1 and x.size == NUM_CLASSES:
        return x.reshape(1, NUM_CLASSES)

    if x.ndim == 2:
        if x.shape[1] != NUM_CLASSES:
            raise ValueError(f"Expected [N, {NUM_CLASSES}], got {x.shape}")
        return x

    if x.ndim == 4:
        # [N, 5, 1, 1]
        if x.shape[1] == NUM_CLASSES and x.shape[2] == 1 and x.shape[3] == 1:
            return x[:, :, 0, 0]

        # [N, 5, L, 1]
        if x.shape[1] == NUM_CLASSES and x.shape[3] == 1:
            return x[:, :, -1, 0]

        # [N, L, 1, 5]
        if x.shape[-1] == NUM_CLASSES and x.shape[2] == 1:
            return x[:, -1, 0, :]

        # [N, 1, L, 5]
        if x.shape[-1] == NUM_CLASSES and x.shape[1] == 1:
            return x[:, 0, -1, :]

    raise ValueError(f"Cannot convert numpy logits with shape {x.shape} to [N, C]")


def logits_to_2d_torch(x):
    x = qt_value(x)

    if x.dim() == 2:
        return x

    if x.dim() == 4:
        # [N, 5, 1, 1]
        if x.shape[1] == NUM_CLASSES and x.shape[2] == 1 and x.shape[3] == 1:
            return x[:, :, 0, 0]

        # [N, 5, L, 1]
        if x.shape[1] == NUM_CLASSES and x.shape[3] == 1:
            return x[:, :, -1, 0]

        # [N, L, 1, 5]
        if x.shape[-1] == NUM_CLASSES and x.shape[2] == 1:
            return x[:, -1, 0, :]

        # [N, 1, L, 5]
        if x.shape[-1] == NUM_CLASSES and x.shape[1] == 1:
            return x[:, 0, -1, :]

    raise ValueError(f"Cannot convert torch logits with shape {tuple(x.shape)} to [N, C]")


# ============================================================
# Load FPGA raw results
# ============================================================

def load_fpga_raw_results():
    if not FPGA_RAW_NPY.exists():
        raise FileNotFoundError(f"FPGA raw logits not found: {FPGA_RAW_NPY}")

    raw = logits_to_2d_np(np.load(str(FPGA_RAW_NPY))).astype(np.float32)

    if FPGA_LABELS_NPY.exists():
        labels = np.load(str(FPGA_LABELS_NPY)).astype(np.int32).reshape(-1)
    else:
        labels = None

    if FPGA_PREDS_NPY.exists():
        preds = np.load(str(FPGA_PREDS_NPY)).astype(np.int32).reshape(-1)
    else:
        preds = raw.argmax(axis=1).astype(np.int32)

    csv_info = read_fpga_indices_from_csv(FPGA_CSV)
    if csv_info is not None:
        indices = csv_info["indices"]
        if labels is None and "labels" in csv_info:
            labels = csv_info["labels"]
        if "preds" in csv_info:
            preds_from_csv = csv_info["preds"]
        else:
            preds_from_csv = preds
    else:
        indices = np.arange(raw.shape[0], dtype=np.int32)
        preds_from_csv = preds
        print("[warn] FPGA CSV with sample indices not found. Assuming first N samples.")

    n = raw.shape[0]
    if MAX_SAMPLES is not None:
        n = min(n, int(MAX_SAMPLES))

    raw = raw[:n]
    indices = indices[:n]
    preds = preds[:n]
    preds_from_csv = preds_from_csv[:n]

    if labels is not None:
        labels = labels[:n]

    return {
        "raw": raw,
        "indices": indices,
        "labels": labels,
        "preds": preds,
        "preds_from_csv": preds_from_csv,
    }


# ============================================================
# Target logits from QONNX or PyTorch
# ============================================================

def pick_qonnx_logits(output_dict, sample_i):
    candidates = []

    for name, val in output_dict.items():
        arr = np.asarray(val)
        try:
            logits = logits_to_2d_np(arr)
            if logits.shape == (1, NUM_CLASSES):
                candidates.append((name, arr, logits))
        except Exception:
            continue

    if len(candidates) == 0:
        print(f"\n[error] Available QONNX outputs for sample {sample_i}:")
        for name, val in output_dict.items():
            print(" ", name, np.asarray(val).shape)
        raise RuntimeError("No logits-like QONNX output found.")

    if sample_i == 0:
        print("\n=== Available QONNX outputs first sample ===")
        for name, val in output_dict.items():
            print(name, np.asarray(val).shape)
        print("Chosen QONNX output:", candidates[0][0], "raw shape:", candidates[0][1].shape)

    return candidates[0][2].astype(np.float32)


def run_qonnx_target(qonnx_path: Path, x_selected):
    if not qonnx_path.exists():
        raise FileNotFoundError(f"QONNX_PATH not found: {qonnx_path}")

    model = ModelWrapper(str(qonnx_path))
    input_name = model.graph.input[0].name

    logits_list = []

    for i in range(x_selected.shape[0]):
        xb = x_selected[i:i + 1].astype(np.float32).reshape(1, 1, SEQ_LEN, 1)
        out_dict = oxe.execute_onnx(model, {input_name: xb})
        logits = pick_qonnx_logits(out_dict, i)
        logits_list.append(logits.reshape(1, NUM_CLASSES))

        if (i + 1) % 100 == 0 or (i + 1) == x_selected.shape[0]:
            print(f"[qonnx] executed {i + 1}/{x_selected.shape[0]} samples")

    return np.concatenate(logits_list, axis=0).astype(np.float32)


def build_pytorch_model_from_checkpoint(ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"CHECKPOINT_PATH not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg = ckpt.get("model_config", {})

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

    model.load_state_dict(clean_sd, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def run_pytorch_target(ckpt_path: Path, x_selected, batch_size=128):
    model = build_pytorch_model_from_checkpoint(ckpt_path)

    logits_list = []
    for start in range(0, x_selected.shape[0], batch_size):
        end = min(start + batch_size, x_selected.shape[0])
        xb = torch.tensor(x_selected[start:end], dtype=torch.float32).unsqueeze(1).unsqueeze(-1)
        xb = xb.to(DEVICE)
        out = model(xb)
        logits = logits_to_2d_torch(out).detach().cpu().numpy().astype(np.float32)
        logits_list.append(logits)

    return np.concatenate(logits_list, axis=0).astype(np.float32)


# ============================================================
# Fitting functions
# ============================================================

def fit_affine_per_class(fpga_raw, target_logits, fit_indices=None):
    """
    Fit per-class affine correction:
        target_logits[:, c] ≈ fpga_raw[:, c] * A[c] + B[c]
    """
    fpga_raw = np.asarray(fpga_raw, dtype=np.float64)
    target_logits = np.asarray(target_logits, dtype=np.float64)

    if fpga_raw.shape != target_logits.shape:
        raise ValueError(f"Shape mismatch: raw {fpga_raw.shape}, target {target_logits.shape}")

    if fit_indices is None:
        fit_indices = np.arange(fpga_raw.shape[0])

    x_fit = fpga_raw[fit_indices]
    y_fit = target_logits[fit_indices]

    A = np.zeros((NUM_CLASSES,), dtype=np.float64)
    B = np.zeros((NUM_CLASSES,), dtype=np.float64)

    for c in range(NUM_CLASSES):
        x = x_fit[:, c]
        y = y_fit[:, c]

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        denom = np.sum((x - x_mean) ** 2)

        if denom < MIN_DENOM:
            A[c] = 0.0
            B[c] = y_mean
            print(f"[warn] class {c}: raw output variance is too small; using A=0, B=mean(target)")
        else:
            A[c] = np.sum((x - x_mean) * (y - y_mean)) / denom
            B[c] = y_mean - A[c] * x_mean

    return A.astype(np.float32), B.astype(np.float32)


def apply_affine(fpga_raw, A, B):
    fpga_raw = np.asarray(fpga_raw, dtype=np.float32)
    return fpga_raw * A.reshape(1, NUM_CLASSES) + B.reshape(1, NUM_CLASSES)


def make_fit_eval_indices(n):
    if FIT_MODE == "all":
        fit_idx = np.arange(n, dtype=np.int32)
        eval_idx = np.arange(n, dtype=np.int32)
    elif FIT_MODE == "holdout":
        split = int(round(n * float(FIT_RATIO)))
        split = max(1, min(n - 1, split))
        fit_idx = np.arange(0, split, dtype=np.int32)
        eval_idx = np.arange(split, n, dtype=np.int32)
    else:
        raise ValueError(f"Unsupported FIT_MODE: {FIT_MODE}")

    return fit_idx, eval_idx


def summarize_subset(name, idx, labels, target_logits, raw_logits, fitted_logits):
    target_pred = target_logits.argmax(axis=1).astype(np.int32)
    raw_pred = raw_logits.argmax(axis=1).astype(np.int32)
    fitted_pred = fitted_logits.argmax(axis=1).astype(np.int32)

    st_raw = diff_stats(raw_logits[idx], target_logits[idx])
    st_fit = diff_stats(fitted_logits[idx], target_logits[idx])

    lines = []
    lines.append(f"\n=== {name} subset ===")
    lines.append(f"samples: {len(idx)}")
    lines.append(f"label distribution: {safe_bincount(labels[idx], NUM_CLASSES).tolist()}")
    lines.append("Accuracy:")
    lines.append(f"  target : {accuracy(target_pred[idx], labels[idx]):.6f}")
    lines.append(f"  raw    : {accuracy(raw_pred[idx], labels[idx]):.6f}")
    lines.append(f"  fitted : {accuracy(fitted_pred[idx], labels[idx]):.6f}")
    lines.append("Prediction agreement with target:")
    lines.append(f"  raw    : {agreement(raw_pred[idx], target_pred[idx]):.6f}")
    lines.append(f"  fitted : {agreement(fitted_pred[idx], target_pred[idx]):.6f}")
    lines.append("Logit diff vs target:")
    lines.append(
        f"  raw    : max={st_raw['max_abs']:.6g}, mean={st_raw['mean_abs']:.6g}, "
        f"median={st_raw['median_abs']:.6g}, p95={st_raw['p95_abs']:.6g}"
    )
    lines.append(
        f"  fitted : max={st_fit['max_abs']:.6g}, mean={st_fit['mean_abs']:.6g}, "
        f"median={st_fit['median_abs']:.6g}, p95={st_fit['p95_abs']:.6g}"
    )

    return lines


def save_compare_csv(indices, labels, target_logits, raw_logits, fitted_logits):
    target_pred = target_logits.argmax(axis=1).astype(np.int32)
    raw_pred = raw_logits.argmax(axis=1).astype(np.int32)
    fitted_pred = fitted_logits.argmax(axis=1).astype(np.int32)

    rows = []
    for i in range(len(labels)):
        row = {
            "row": i,
            "sample_index": int(indices[i]),
            "label": int(labels[i]),
            "target_pred": int(target_pred[i]),
            "raw_pred": int(raw_pred[i]),
            "fitted_pred": int(fitted_pred[i]),
            "target_correct": int(target_pred[i] == labels[i]),
            "raw_correct": int(raw_pred[i] == labels[i]),
            "fitted_correct": int(fitted_pred[i] == labels[i]),
            "raw_target_match": int(raw_pred[i] == target_pred[i]),
            "fitted_target_match": int(fitted_pred[i] == target_pred[i]),
        }

        for c in range(NUM_CLASSES):
            row[f"target_logit_{c}"] = float(target_logits[i, c])
            row[f"raw_logit_{c}"] = float(raw_logits[i, c])
            row[f"fitted_logit_{c}"] = float(fitted_logits[i, c])

        rows.append(row)

    with open(OUT_COMPARE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("[info] Saved compare CSV:", OUT_COMPARE_CSV)


def save_outputs(A, B, metadata, target_logits, raw_logits, fitted_logits):
    FPGA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(str(OUT_NPY_A), A)
    np.save(str(OUT_NPY_B), B)

    payload = {
        "OUTPUT_A": A.tolist(),
        "OUTPUT_B": B.tolist(),
        "metadata": metadata,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    snippet = """
# ============================================================
# Fitted FPGA output affine correction
# ============================================================
APPLY_OUTPUT_AFFINE = True

OUTPUT_A = np.array(
    {A_list},
    dtype=np.float32,
)
OUTPUT_B = np.array(
    {B_list},
    dtype=np.float32,
)


def postprocess_logits(logits_raw):
    logits_raw = np.asarray(logits_raw, dtype=np.float32).reshape(NUM_CLASSES)
    if APPLY_OUTPUT_AFFINE:
        return logits_raw * OUTPUT_A + OUTPUT_B
    return logits_raw
""".strip().format(
        A_list=repr(A.tolist()),
        B_list=repr(B.tolist()),
    )

    with open(OUT_TXT, "w") as f:
        f.write(snippet + "\n")

    print("[info] Saved:")
    print(" ", OUT_JSON)
    print(" ", OUT_NPY_A)
    print(" ", OUT_NPY_B)
    print(" ", OUT_TXT)


# ============================================================
# Main
# ============================================================

def main():
    print("===================================================")
    print("Fit FPGA output affine correction")
    print("===================================================")
    print("THIS_DIR        :", THIS_DIR)
    print("PROJECT_DIR     :", PROJECT_DIR)
    print("CALIB_TXT       :", CALIB_TXT)
    print("QONNX_PATH      :", QONNX_PATH)
    print("CHECKPOINT_PATH :", CHECKPOINT_PATH)
    print("FPGA_RESULTS_DIR:", FPGA_RESULTS_DIR)
    print("FPGA_RAW_NPY    :", FPGA_RAW_NPY)
    print("TARGET_SOURCE   :", TARGET_SOURCE)
    print("FIT_MODE        :", FIT_MODE)

    fpga = load_fpga_raw_results()
    fpga_raw = fpga["raw"]
    indices = fpga["indices"]
    fpga_labels = fpga["labels"]

    x_all, y_all = load_ecg_txt(CALIB_TXT)

    if np.max(indices) >= len(x_all):
        raise ValueError(
            f"FPGA sample index {np.max(indices)} exceeds CALIB_TXT length {len(x_all)}.\n"
            "This usually means CALIB_TXT does not match the dataset used on the board."
        )

    x_sel = x_all[indices]
    labels = y_all[indices]

    if fpga_labels is not None and not np.array_equal(labels, fpga_labels):
        print("[warn] Labels from CALIB_TXT do not match FPGA labels file.")
        print("       CALIB label distribution:", safe_bincount(labels, NUM_CLASSES).tolist())
        print("       FPGA  label distribution:", safe_bincount(fpga_labels, NUM_CLASSES).tolist())
        print("       Using FPGA labels for reported accuracy.")
        labels = fpga_labels

    print("\n[info] Loaded samples:", len(labels))
    print("[info] FPGA raw shape:", fpga_raw.shape)
    print("[info] indices first 20:", indices[:20])
    print("[info] label distribution:", safe_bincount(labels, NUM_CLASSES).tolist())
    print("[info] FPGA raw first row:", fpga_raw[0])

    if TARGET_SOURCE.lower() == "qonnx":
        print("\n=== Running QONNX target ===")
        target_logits = run_qonnx_target(QONNX_PATH, x_sel)
    elif TARGET_SOURCE.lower() == "pytorch":
        print("\n=== Running PyTorch target ===")
        target_logits = run_pytorch_target(CHECKPOINT_PATH, x_sel)
    else:
        raise ValueError(f"Unsupported TARGET_SOURCE: {TARGET_SOURCE}")

    print("[info] target logits shape:", target_logits.shape)
    print("[info] target first row:", target_logits[0])

    if fpga_raw.shape != target_logits.shape:
        raise ValueError(f"Shape mismatch: FPGA raw {fpga_raw.shape}, target {target_logits.shape}")

    n = len(labels)
    fit_idx, eval_idx = make_fit_eval_indices(n)

    print("\n[info] fit samples :", len(fit_idx))
    print("[info] eval samples:", len(eval_idx))

    A, B = fit_affine_per_class(fpga_raw, target_logits, fit_indices=fit_idx)
    fitted_logits = apply_affine(fpga_raw, A, B)

    print("\n===================================================")
    print("Fitted affine correction")
    print("===================================================")
    print("OUTPUT_A = np.array(")
    print("    " + repr(A.tolist()) + ",")
    print("    dtype=np.float32,")
    print(")")
    print("OUTPUT_B = np.array(")
    print("    " + repr(B.tolist()) + ",")
    print("    dtype=np.float32,")
    print(")")

    summary_lines = []
    summary_lines += summarize_subset("FIT", fit_idx, labels, target_logits, fpga_raw, fitted_logits)
    summary_lines += summarize_subset("EVAL", eval_idx, labels, target_logits, fpga_raw, fitted_logits)

    print("\n" + "\n".join(summary_lines))

    target_pred = target_logits.argmax(axis=1).astype(np.int32)
    raw_pred = fpga_raw.argmax(axis=1).astype(np.int32)
    fitted_pred = fitted_logits.argmax(axis=1).astype(np.int32)

    metadata = {
        "target_source": TARGET_SOURCE,
        "fit_mode": FIT_MODE,
        "fit_ratio": FIT_RATIO,
        "num_samples": int(n),
        "num_fit_samples": int(len(fit_idx)),
        "num_eval_samples": int(len(eval_idx)),
        "calib_txt": str(CALIB_TXT),
        "qonnx_path": str(QONNX_PATH),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "fpga_results_dir": str(FPGA_RESULTS_DIR),
        "raw_accuracy_all": accuracy(raw_pred, labels),
        "fitted_accuracy_all": accuracy(fitted_pred, labels),
        "target_accuracy_all": accuracy(target_pred, labels),
        "raw_target_agreement_all": agreement(raw_pred, target_pred),
        "fitted_target_agreement_all": agreement(fitted_pred, target_pred),
        "raw_diff_stats_all": diff_stats(fpga_raw, target_logits),
        "fitted_diff_stats_all": diff_stats(fitted_logits, target_logits),
    }

    save_outputs(A, B, metadata, target_logits, fpga_raw, fitted_logits)
    save_compare_csv(indices, labels, target_logits, fpga_raw, fitted_logits)

    print("\n=== First 10 sample detail ===")
    for i in range(min(10, n)):
        print(
            f"i={i} idx={int(indices[i])} label={int(labels[i])} "
            f"target={int(target_pred[i])} raw={int(raw_pred[i])} fitted={int(fitted_pred[i])}"
        )
        print("  raw    :", fpga_raw[i])
        print("  target :", target_logits[i])
        print("  fitted :", fitted_logits[i])

    print("\nDone.")
    print("Paste the printed OUTPUT_A / OUTPUT_B into your PYNQ test script.")
    print("For final thesis results, fit these on validation/calibration samples, not on the final test set.")


if __name__ == "__main__":
    main()

