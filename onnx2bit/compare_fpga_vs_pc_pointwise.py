from pathlib import Path
import numpy as np


# ============================================================
# Paths
# ============================================================

THIS_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = THIS_DIR.parent
PROJECT_DIR = NOTEBOOKS_DIR / "pytorch-tcn"

# PC 端验证脚本 verify_pointwise_export.py 保存的输出
PC_DIR = PROJECT_DIR / "verify_pointwise_outputs"

# 从 PYNQ 板子拷回来的结果
FPGA_DIR = PROJECT_DIR / "board_results_pointwise"

OUT_DIR = PROJECT_DIR / "compare_fpga_vs_pc_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Basic config
# ============================================================

NUM_CLASSES = 5
SEQ_LEN = 140

# 如果 PC verify_pointwise_export.py 只跑了前 100 个样本，而 FPGA 跑了 4500，
# 这里会自动检测长度不一致并提示。
STRICT_LENGTH_MATCH = False

PRINT_FIRST_N = 20


# ============================================================
# File paths
# ============================================================

FPGA_LOGITS_PATH = FPGA_DIR / "fpga_last_step_logits.npy"
FPGA_PREDS_PATH = FPGA_DIR / "fpga_preds.npy"
FPGA_LABELS_PATH = FPGA_DIR / "ecg5000_test_labels.npy"
FPGA_ALL_OUTPUTS_PATH = FPGA_DIR / "fpga_all_time_outputs.npy"

PC_LABELS_PATH = PC_DIR / "labels.npy"
PC_INDICES_PATH = PC_DIR / "selected_indices.npy"

PYTORCH_LOGITS_PATH = PC_DIR / "pytorch_last_logits.npy"
PYTORCH_PREDS_PATH = PC_DIR / "pytorch_preds.npy"
PYTORCH_RAW_PATH = PC_DIR / "pytorch_raw_all_time_logits.npy"

CLEAN_LOGITS_PATH = PC_DIR / "clean_qonnx_last_logits.npy"
CLEAN_PREDS_PATH = PC_DIR / "clean_qonnx_preds.npy"
CLEAN_RAW_PATH = PC_DIR / "clean_qonnx_raw_all_time_logits.npy"

FINN_LOGITS_PATH = PC_DIR / "finn_streamlined_last_logits.npy"
FINN_PREDS_PATH = PC_DIR / "finn_streamlined_preds.npy"
FINN_RAW_PATH = PC_DIR / "finn_streamlined_raw_all_time_logits.npy"


# ============================================================
# Helpers
# ============================================================

def load_required(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    arr = np.load(str(path), allow_pickle=True)
    print(f"[load] {name}: {path}")
    print(f"       shape={arr.shape}, dtype={arr.dtype}")
    return arr


def load_optional(path: Path, name: str):
    if not path.exists():
        print(f"[skip] {name} not found: {path}")
        return None
    arr = np.load(str(path), allow_pickle=True)
    print(f"[load] {name}: {path}")
    print(f"       shape={arr.shape}, dtype={arr.dtype}")
    return arr


def as_int_1d(x):
    return np.asarray(x).astype(np.int32).reshape(-1)


def as_float_2d(x):
    x = np.asarray(x).astype(np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D logits [N, C], got {x.shape}")
    return x


def safe_bincount(x, minlength=5):
    x = as_int_1d(x)
    x = x[x >= 0]
    if x.size == 0:
        return np.zeros((minlength,), dtype=np.int32)
    return np.bincount(x, minlength=minlength)


def accuracy(preds, labels):
    preds = as_int_1d(preds)
    labels = as_int_1d(labels)
    return float(np.mean(preds == labels))


def confusion_matrix(labels, preds, num_classes=5):
    labels = as_int_1d(labels)
    preds = as_int_1d(preds)

    mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    for y, p in zip(labels, preds):
        if 0 <= y < num_classes and 0 <= p < num_classes:
            mat[y, p] += 1

    return mat


def print_confusion_matrix(labels, preds, title):
    mat = confusion_matrix(labels, preds, NUM_CLASSES)

    print(f"\n=== Confusion matrix: {title} ===")
    print("rows = true label, cols = pred")
    print(mat)

    return mat


def print_per_class(labels, preds, title):
    labels = as_int_1d(labels)
    preds = as_int_1d(preds)

    print(f"\n=== Per-class accuracy: {title} ===")

    rows = []

    for c in range(NUM_CLASSES):
        mask = labels == c
        total = int(np.sum(mask))
        correct = int(np.sum(preds[mask] == labels[mask])) if total > 0 else 0
        acc = correct / total if total > 0 else 0.0
        rows.append((c, correct, total, acc))
        print(f"class {c}: {correct}/{total}, acc={acc:.4f}")

    return rows


def summarize_preds(labels, preds, title):
    labels = as_int_1d(labels)
    preds = as_int_1d(preds)

    print(f"\n===================================================")
    print(title)
    print("===================================================")
    print("num samples:", len(labels))
    print("accuracy   :", accuracy(preds, labels))
    print("label distribution:", safe_bincount(labels, NUM_CLASSES))
    print("pred  distribution:", safe_bincount(preds, NUM_CLASSES))

    print_per_class(labels, preds, title)
    print_confusion_matrix(labels, preds, title)


def compare_preds(labels, ref_preds, other_preds, ref_name, other_name):
    labels = as_int_1d(labels)
    ref_preds = as_int_1d(ref_preds)
    other_preds = as_int_1d(other_preds)

    print(f"\n===================================================")
    print(f"Prediction comparison: {ref_name} vs {other_name}")
    print("===================================================")

    print(f"{ref_name} acc:", accuracy(ref_preds, labels))
    print(f"{other_name} acc:", accuracy(other_preds, labels))
    print("pred match:", float(np.mean(ref_preds == other_preds)))

    mismatch = np.where(ref_preds != other_preds)[0]
    print("num pred mismatch:", len(mismatch))

    if len(mismatch) > 0:
        print("first mismatches:")
        for i in mismatch[:PRINT_FIRST_N]:
            print(
                f"i={i} label={int(labels[i])} "
                f"{ref_name}_pred={int(ref_preds[i])} "
                f"{other_name}_pred={int(other_preds[i])}"
            )


def fit_global_scale(ref_float_logits, fpga_int_logits):
    """
    Fit one global scale alpha such that:
        ref_float ≈ alpha * fpga_int

    This is useful because FPGA driver often returns raw integer-domain INT24 logits,
    while PC FINN/QONNX may return scaled float logits.

    Argmax is invariant to positive global scale, so prediction agreement matters more
    than raw absolute diff.
    """
    ref = np.asarray(ref_float_logits).astype(np.float64).reshape(-1)
    x = np.asarray(fpga_int_logits).astype(np.float64).reshape(-1)

    denom = float(np.sum(x * x))

    if denom == 0:
        return 0.0

    alpha = float(np.sum(ref * x) / denom)
    return alpha


def compare_logits(ref_name, ref_logits, fpga_logits, labels=None, ref_preds=None, fpga_preds=None):
    ref_logits = as_float_2d(ref_logits)
    fpga_logits = as_float_2d(fpga_logits)

    if ref_logits.shape != fpga_logits.shape:
        print(f"\n[skip] Logit compare {ref_name} vs FPGA: shape mismatch")
        print("ref :", ref_logits.shape)
        print("fpga:", fpga_logits.shape)
        return None

    print(f"\n===================================================")
    print(f"Logit comparison: {ref_name} float logits vs FPGA raw logits")
    print("===================================================")

    raw_diff = fpga_logits - ref_logits

    print("shape:", ref_logits.shape)
    print("raw FPGA range:", float(fpga_logits.min()), float(fpga_logits.max()))
    print(f"{ref_name} range:", float(ref_logits.min()), float(ref_logits.max()))

    print("\nRaw direct diff, usually not meaningful if domains differ:")
    print("max abs diff :", float(np.max(np.abs(raw_diff))))
    print("mean abs diff:", float(np.mean(np.abs(raw_diff))))

    alpha = fit_global_scale(ref_logits, fpga_logits)

    scaled_fpga = fpga_logits * alpha
    scaled_diff = scaled_fpga - ref_logits

    print("\nBest global scale:")
    print("alpha such that ref ≈ alpha * fpga:", alpha)

    if alpha > 0:
        print("[ok] positive scale, argmax should be preserved by scaling.")
    else:
        print("[warn] non-positive scale. Something may be wrong.")

    print("\nAfter global scaling:")
    print("scaled FPGA range:", float(scaled_fpga.min()), float(scaled_fpga.max()))
    print("max abs diff :", float(np.max(np.abs(scaled_diff))))
    print("mean abs diff:", float(np.mean(np.abs(scaled_diff))))

    # Correlation
    ref_flat = ref_logits.reshape(-1).astype(np.float64)
    fpga_flat = fpga_logits.reshape(-1).astype(np.float64)

    if np.std(ref_flat) > 0 and np.std(fpga_flat) > 0:
        corr = float(np.corrcoef(ref_flat, fpga_flat)[0, 1])
    else:
        corr = float("nan")

    print("global correlation:", corr)

    if ref_preds is None:
        ref_preds = np.argmax(ref_logits, axis=1).astype(np.int32)
    else:
        ref_preds = as_int_1d(ref_preds)

    if fpga_preds is None:
        fpga_preds = np.argmax(fpga_logits, axis=1).astype(np.int32)
    else:
        fpga_preds = as_int_1d(fpga_preds)

    print("\nPrediction-level:")
    print("pred match:", float(np.mean(ref_preds == fpga_preds)))

    if labels is not None:
        labels = as_int_1d(labels)
        print(f"{ref_name} acc:", accuracy(ref_preds, labels))
        print("FPGA acc:", accuracy(fpga_preds, labels))

    print("\nFirst samples:")
    for i in range(min(PRINT_FIRST_N, ref_logits.shape[0])):
        print(f"i={i}")
        if labels is not None:
            print(" label:", int(labels[i]))
        print(f" {ref_name} pred:", int(ref_preds[i]), "logits:", ref_logits[i])
        print(" FPGA pred:", int(fpga_preds[i]), "raw:", fpga_logits[i])
        print(" scaled FPGA:", scaled_fpga[i])
        print(" scaled diff:", scaled_diff[i])

    return {
        "alpha": alpha,
        "corr": corr,
        "raw_max_abs_diff": float(np.max(np.abs(raw_diff))),
        "raw_mean_abs_diff": float(np.mean(np.abs(raw_diff))),
        "scaled_max_abs_diff": float(np.max(np.abs(scaled_diff))),
        "scaled_mean_abs_diff": float(np.mean(np.abs(scaled_diff))),
        "pred_match": float(np.mean(ref_preds == fpga_preds)),
    }


def canonical_all_time_logits(x, name):
    """
    Convert all-time logits into [N, 140, 5].

    Supported:
        FPGA: [N, 1, 140, 1, 5]
        PC:   [N, 5, 140, 1]
        PC:   [N, 140, 1, 5]
        PC:   [N, 1, 140, 5]
    """
    x = np.asarray(x).astype(np.float32)

    # FPGA saved shape: [N, 1, 140, 1, 5]
    if x.ndim == 5:
        if x.shape[1] == 1 and x.shape[2] == SEQ_LEN and x.shape[3] == 1 and x.shape[4] == NUM_CLASSES:
            return x[:, 0, :, 0, :]

    if x.ndim == 4:
        # [N, 5, 140, 1]
        if x.shape[1] == NUM_CLASSES and x.shape[2] == SEQ_LEN and x.shape[3] == 1:
            return np.transpose(x[:, :, :, 0], (0, 2, 1))

        # [N, 140, 1, 5]
        if x.shape[1] == SEQ_LEN and x.shape[2] == 1 and x.shape[3] == NUM_CLASSES:
            return x[:, :, 0, :]

        # [N, 1, 140, 5]
        if x.shape[1] == 1 and x.shape[2] == SEQ_LEN and x.shape[3] == NUM_CLASSES:
            return x[:, 0, :, :]

    raise ValueError(f"{name}: unsupported all-time logits shape {x.shape}")


def compare_all_time(ref_name, ref_raw, fpga_raw):
    if ref_raw is None or fpga_raw is None:
        print(f"[skip] all-time compare for {ref_name}: missing raw outputs")
        return None

    try:
        ref = canonical_all_time_logits(ref_raw, ref_name)
        fpga = canonical_all_time_logits(fpga_raw, "FPGA")
    except Exception as e:
        print(f"[skip] all-time compare for {ref_name}: {e}")
        return None

    if ref.shape != fpga.shape:
        print(f"[skip] all-time compare {ref_name} vs FPGA: shape mismatch")
        print("ref :", ref.shape)
        print("fpga:", fpga.shape)
        return None

    print(f"\n===================================================")
    print(f"All-time output comparison: {ref_name} vs FPGA")
    print("===================================================")
    print("canonical shape [N, 140, 5]:", ref.shape)

    alpha = fit_global_scale(ref, fpga)
    scaled_fpga = fpga * alpha

    raw_diff = fpga - ref
    scaled_diff = scaled_fpga - ref

    print("raw fpga range:", float(fpga.min()), float(fpga.max()))
    print(f"{ref_name} range:", float(ref.min()), float(ref.max()))

    print("\nRaw direct diff:")
    print("max abs diff :", float(np.max(np.abs(raw_diff))))
    print("mean abs diff:", float(np.mean(np.abs(raw_diff))))

    print("\nBest global scale:")
    print("alpha:", alpha)

    print("\nAfter scaling:")
    print("max abs diff :", float(np.max(np.abs(scaled_diff))))
    print("mean abs diff:", float(np.mean(np.abs(scaled_diff))))

    ref_last = ref[:, -1, :]
    fpga_last = fpga[:, -1, :]
    ref_pred = np.argmax(ref_last, axis=1).astype(np.int32)
    fpga_pred = np.argmax(fpga_last, axis=1).astype(np.int32)

    print("last-step pred match:", float(np.mean(ref_pred == fpga_pred)))

    return {
        "alpha": alpha,
        "raw_max_abs_diff": float(np.max(np.abs(raw_diff))),
        "raw_mean_abs_diff": float(np.mean(np.abs(raw_diff))),
        "scaled_max_abs_diff": float(np.max(np.abs(scaled_diff))),
        "scaled_mean_abs_diff": float(np.mean(np.abs(scaled_diff))),
        "last_step_pred_match": float(np.mean(ref_pred == fpga_pred)),
    }


def maybe_align_pc_to_fpga(fpga_labels, pc_labels, pc_indices, arrays):
    """
    If PC verification was run on selected subset, try to align.

    arrays is dict name -> np array with first dimension N.
    """
    fpga_labels = as_int_1d(fpga_labels)

    if pc_labels is None:
        return arrays, None

    pc_labels = as_int_1d(pc_labels)

    if len(pc_labels) == len(fpga_labels):
        return arrays, pc_labels

    print("\n[warn] PC and FPGA lengths differ.")
    print("FPGA N:", len(fpga_labels))
    print("PC N  :", len(pc_labels))

    if pc_indices is None:
        if STRICT_LENGTH_MATCH:
            raise RuntimeError("Length mismatch and no selected_indices.npy found.")
        print("[warn] Cannot align because selected_indices.npy is missing.")
        return arrays, pc_labels

    pc_indices = as_int_1d(pc_indices)

    print("PC selected_indices length:", len(pc_indices))

    # Common case:
    # PC verify used first N samples. FPGA full results.
    # Then we can compare FPGA subset pc_indices.
    if np.max(pc_indices) < len(fpga_labels):
        print("[info] Will compare FPGA subset using PC selected_indices.")

        aligned = {}
        for name, arr in arrays.items():
            if arr is None:
                aligned[name] = None
                continue

            arr = np.asarray(arr)

            if arr.shape[0] == len(fpga_labels):
                aligned[name] = arr[pc_indices]
            else:
                aligned[name] = arr

        return aligned, fpga_labels[pc_indices]

    if STRICT_LENGTH_MATCH:
        raise RuntimeError("Length mismatch and selected_indices cannot align.")

    print("[warn] selected_indices cannot align to FPGA results.")
    return arrays, pc_labels


def save_summary(summary):
    path = OUT_DIR / "compare_summary.txt"

    with open(path, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("\n[info] Summary saved to:", path)


# ============================================================
# Main
# ============================================================

def main():
    print("===================================================")
    print("Compare FPGA pointwise outputs vs PC reference")
    print("===================================================")
    print("PROJECT_DIR:", PROJECT_DIR)
    print("PC_DIR     :", PC_DIR)
    print("FPGA_DIR   :", FPGA_DIR)
    print("OUT_DIR    :", OUT_DIR)

    # ------------------------------------------------------------
    # Load FPGA results
    # ------------------------------------------------------------
    fpga_logits = as_float_2d(load_required(FPGA_LOGITS_PATH, "FPGA last-step logits"))
    fpga_preds = as_int_1d(load_required(FPGA_PREDS_PATH, "FPGA preds"))
    fpga_labels = as_int_1d(load_required(FPGA_LABELS_PATH, "FPGA labels"))
    fpga_raw = load_optional(FPGA_ALL_OUTPUTS_PATH, "FPGA all-time outputs")

    if not (len(fpga_logits) == len(fpga_preds) == len(fpga_labels)):
        raise RuntimeError(
            "FPGA result length mismatch:\n"
            f"logits={len(fpga_logits)}, preds={len(fpga_preds)}, labels={len(fpga_labels)}"
        )

    summarize_preds(fpga_labels, fpga_preds, "FPGA board result")

    # ------------------------------------------------------------
    # Load PC references
    # ------------------------------------------------------------
    pc_labels = load_optional(PC_LABELS_PATH, "PC labels")
    if pc_labels is not None:
        pc_labels = as_int_1d(pc_labels)

    pc_indices = load_optional(PC_INDICES_PATH, "PC selected indices")
    if pc_indices is not None:
        pc_indices = as_int_1d(pc_indices)

    pytorch_logits = load_optional(PYTORCH_LOGITS_PATH, "PyTorch last-step logits")
    pytorch_preds = load_optional(PYTORCH_PREDS_PATH, "PyTorch preds")
    pytorch_raw = load_optional(PYTORCH_RAW_PATH, "PyTorch all-time raw")

    clean_logits = load_optional(CLEAN_LOGITS_PATH, "CLEAN_QONNX last-step logits")
    clean_preds = load_optional(CLEAN_PREDS_PATH, "CLEAN_QONNX preds")
    clean_raw = load_optional(CLEAN_RAW_PATH, "CLEAN_QONNX all-time raw")

    finn_logits = load_optional(FINN_LOGITS_PATH, "FINN_STREAMLINED last-step logits")
    finn_preds = load_optional(FINN_PREDS_PATH, "FINN_STREAMLINED preds")
    finn_raw = load_optional(FINN_RAW_PATH, "FINN_STREAMLINED all-time raw")

    # Convert loaded arrays
    if pytorch_logits is not None:
        pytorch_logits = as_float_2d(pytorch_logits)
    if pytorch_preds is not None:
        pytorch_preds = as_int_1d(pytorch_preds)

    if clean_logits is not None:
        clean_logits = as_float_2d(clean_logits)
    if clean_preds is not None:
        clean_preds = as_int_1d(clean_preds)

    if finn_logits is not None:
        finn_logits = as_float_2d(finn_logits)
    if finn_preds is not None:
        finn_preds = as_int_1d(finn_preds)

    # ------------------------------------------------------------
    # Align FPGA to PC subset if needed
    # ------------------------------------------------------------
    arrays = {
        "fpga_logits": fpga_logits,
        "fpga_preds": fpga_preds,
        "fpga_raw": fpga_raw,
    }

    arrays, labels_for_pc_compare = maybe_align_pc_to_fpga(
        fpga_labels=fpga_labels,
        pc_labels=pc_labels,
        pc_indices=pc_indices,
        arrays=arrays,
    )

    fpga_logits_aligned = arrays["fpga_logits"]
    fpga_preds_aligned = as_int_1d(arrays["fpga_preds"])
    fpga_raw_aligned = arrays["fpga_raw"]

    if labels_for_pc_compare is None:
        labels_for_pc_compare = fpga_labels

    labels_for_pc_compare = as_int_1d(labels_for_pc_compare)

    # ------------------------------------------------------------
    # Summaries for PC refs
    # ------------------------------------------------------------
    summary = {}

    summary["fpga_acc"] = accuracy(fpga_preds, fpga_labels)
    summary["fpga_pred_distribution"] = safe_bincount(fpga_preds, NUM_CLASSES).tolist()

    refs = [
        ("PyTorch", pytorch_logits, pytorch_preds, pytorch_raw),
        ("CLEAN_QONNX", clean_logits, clean_preds, clean_raw),
        ("FINN_STREAMLINED", finn_logits, finn_preds, finn_raw),
    ]

    for name, logits, preds, raw in refs:
        if logits is None:
            continue

        if preds is None:
            preds = np.argmax(logits, axis=1).astype(np.int32)

        if len(preds) != len(labels_for_pc_compare):
            print(f"\n[skip] {name} length mismatch after alignment:")
            print(f"{name} N:", len(preds))
            print("labels N:", len(labels_for_pc_compare))
            continue

        summarize_preds(labels_for_pc_compare, preds, name)

        compare_preds(
            labels=labels_for_pc_compare,
            ref_preds=preds,
            other_preds=fpga_preds_aligned,
            ref_name=name,
            other_name="FPGA",
        )

        logit_stats = compare_logits(
            ref_name=name,
            ref_logits=logits,
            fpga_logits=fpga_logits_aligned,
            labels=labels_for_pc_compare,
            ref_preds=preds,
            fpga_preds=fpga_preds_aligned,
        )

        all_time_stats = compare_all_time(
            ref_name=name,
            ref_raw=raw,
            fpga_raw=fpga_raw_aligned,
        )

        summary[f"{name}_acc"] = accuracy(preds, labels_for_pc_compare)
        summary[f"{name}_vs_fpga_pred_match"] = float(np.mean(preds == fpga_preds_aligned))

        if logit_stats is not None:
            for k, v in logit_stats.items():
                summary[f"{name}_last_logits_{k}"] = v

        if all_time_stats is not None:
            for k, v in all_time_stats.items():
                summary[f"{name}_all_time_{k}"] = v

    # ------------------------------------------------------------
    # Save aligned data for further inspection
    # ------------------------------------------------------------
    np.save(str(OUT_DIR / "fpga_logits_aligned.npy"), fpga_logits_aligned.astype(np.float32))
    np.save(str(OUT_DIR / "fpga_preds_aligned.npy"), fpga_preds_aligned.astype(np.int32))
    np.save(str(OUT_DIR / "labels_for_compare.npy"), labels_for_pc_compare.astype(np.int32))

    save_summary(summary)

    print("\n===================================================")
    print("Done")
    print("===================================================")
    print("Important notes:")
    print("1. FPGA logits are likely raw INT24 accumulator values.")
    print("2. PC FINN/QONNX logits may be scaled float values.")
    print("3. Therefore raw absolute diff can be huge and is not always meaningful.")
    print("4. Focus on pred match, accuracy, and positive global scale correlation.")
    print("5. If PC verification was not full 4500 samples, rerun verify_pointwise_export.py with:")
    print("   MAX_SAMPLES = None")
    print("   DEBUG_BALANCED_PER_CLASS = None")


if __name__ == "__main__":
    main()