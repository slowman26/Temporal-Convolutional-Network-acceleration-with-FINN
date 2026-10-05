from pathlib import Path
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = THIS_DIR.parent
PROJECT_DIR = NOTEBOOKS_DIR / "pytorch-tcn"

PC_DIR = PROJECT_DIR / "verify_pointwise_outputs"
FPGA_DIR = PROJECT_DIR / "board_results_pointwise"

NUM_CLASSES = 5


def acc(pred, label):
    return float(np.mean(pred.astype(np.int32) == label.astype(np.int32)))


def safe_bincount(x, minlength=5):
    x = np.asarray(x).astype(np.int32).reshape(-1)
    return np.bincount(x, minlength=minlength)


def fit_global_affine(x, y):
    """
    Fit:
        y ≈ a * x + b
    using all classes together.
    """
    x = x.reshape(-1).astype(np.float64)
    y = y.reshape(-1).astype(np.float64)

    A = np.stack([x, np.ones_like(x)], axis=1)
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def fit_per_class_affine(x, y):
    """
    Fit per class:
        y[:, c] ≈ a[c] * x[:, c] + b[c]
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    a = np.zeros((x.shape[1],), dtype=np.float64)
    b = np.zeros((x.shape[1],), dtype=np.float64)

    for c in range(x.shape[1]):
        A = np.stack([x[:, c], np.ones_like(x[:, c])], axis=1)
        a[c], b[c] = np.linalg.lstsq(A, y[:, c], rcond=None)[0]

    return a.astype(np.float32), b.astype(np.float32)


def main():
    fpga_logits = np.load(FPGA_DIR / "fpga_last_step_logits.npy").astype(np.float32)
    fpga_preds = np.load(FPGA_DIR / "fpga_preds.npy").astype(np.int32)
    labels = np.load(FPGA_DIR / "ecg5000_test_labels.npy").astype(np.int32)

    finn_logits = np.load(PC_DIR / "finn_streamlined_last_logits.npy").astype(np.float32)
    finn_preds = np.load(PC_DIR / "finn_streamlined_preds.npy").astype(np.int32)

    pc_labels = np.load(PC_DIR / "labels.npy").astype(np.int32)

    n = min(len(fpga_logits), len(finn_logits))

    fpga_logits = fpga_logits[:n]
    fpga_preds = fpga_preds[:n]
    labels = labels[:n]

    finn_logits = finn_logits[:n]
    finn_preds = finn_preds[:n]
    pc_labels = pc_labels[:n]

    if not np.array_equal(labels, pc_labels):
        print("[warn] FPGA labels and PC labels are not identical for compared subset.")
        print("FPGA labels first 20:", labels[:20])
        print("PC labels first 20  :", pc_labels[:20])

    print("===================================================")
    print("Raw comparison")
    print("===================================================")
    print("N:", n)
    print("label distribution:", safe_bincount(labels, NUM_CLASSES))
    print("FINN acc:", acc(finn_preds, labels))
    print("FPGA acc:", acc(fpga_preds, labels))
    print("raw pred match:", float(np.mean(finn_preds == fpga_preds)))
    print("FINN pred distribution:", safe_bincount(finn_preds, NUM_CLASSES))
    print("FPGA pred distribution:", safe_bincount(fpga_preds, NUM_CLASSES))

    # ------------------------------------------------------------
    # Global scale only
    # ------------------------------------------------------------
    denom = np.sum(fpga_logits.astype(np.float64) ** 2)
    scale = float(np.sum(finn_logits.astype(np.float64) * fpga_logits.astype(np.float64)) / denom)

    fpga_global_scaled = fpga_logits * scale
    pred_global_scaled = np.argmax(fpga_global_scaled, axis=1).astype(np.int32)

    print("\n===================================================")
    print("Global scale only")
    print("===================================================")
    print("scale:", scale)
    print("pred match:", float(np.mean(pred_global_scaled == finn_preds)))
    print("scaled FPGA acc:", acc(pred_global_scaled, labels))
    print("mean abs diff:", float(np.mean(np.abs(fpga_global_scaled - finn_logits))))
    print("max abs diff :", float(np.max(np.abs(fpga_global_scaled - finn_logits))))

    # ------------------------------------------------------------
    # Global affine
    # ------------------------------------------------------------
    a_g, b_g = fit_global_affine(fpga_logits, finn_logits)
    fpga_global_affine = fpga_logits * a_g + b_g
    pred_global_affine = np.argmax(fpga_global_affine, axis=1).astype(np.int32)

    print("\n===================================================")
    print("Global affine: y ≈ a * raw + b")
    print("===================================================")
    print("a:", a_g)
    print("b:", b_g)
    print("pred match:", float(np.mean(pred_global_affine == finn_preds)))
    print("global-affine FPGA acc:", acc(pred_global_affine, labels))
    print("mean abs diff:", float(np.mean(np.abs(fpga_global_affine - finn_logits))))
    print("max abs diff :", float(np.max(np.abs(fpga_global_affine - finn_logits))))

    # ------------------------------------------------------------
    # Per-class affine
    # ------------------------------------------------------------
    a_c, b_c = fit_per_class_affine(fpga_logits, finn_logits)
    fpga_class_affine = fpga_logits * a_c.reshape(1, -1) + b_c.reshape(1, -1)
    pred_class_affine = np.argmax(fpga_class_affine, axis=1).astype(np.int32)

    print("\n===================================================")
    print("Per-class affine: y[:,c] ≈ a[c] * raw[:,c] + b[c]")
    print("===================================================")
    print("a_c:", a_c)
    print("b_c:", b_c)
    print("pred match:", float(np.mean(pred_class_affine == finn_preds)))
    print("per-class-affine FPGA acc:", acc(pred_class_affine, labels))
    print("mean abs diff:", float(np.mean(np.abs(fpga_class_affine - finn_logits))))
    print("max abs diff :", float(np.max(np.abs(fpga_class_affine - finn_logits))))

    print("\nFirst 20 comparison:")
    for i in range(min(20, n)):
        print(f"i={i}, label={labels[i]}")
        print("  FINN pred:", finn_preds[i], "FINN logits:", finn_logits[i])
        print("  FPGA raw pred:", fpga_preds[i], "raw:", fpga_logits[i])
        print("  FPGA per-class-affine pred:", pred_class_affine[i], "logits:", fpga_class_affine[i])


if __name__ == "__main__":
    main()