import copy
import importlib
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# 1. Hyperparameters
# =========================================================
NUM_CLASSES = 5
SEQ_LEN = 140
NUM_INPUTS = 1
NUM_CHANNELS = (16, 16, 16)
KERNEL_SIZE = 5
DROPOUT = 0.1
CAUSAL = True

BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10

USE_CLASS_WEIGHT = True
SEED = 42

# This version is for the 3D feature model:
#     input  : [N, 1, 140]
#     feature: [N, C, L]
#     head   : Flatten + QuantLinear
#     output : [N, 5]
MODEL_MODULE_NAME = "newnet_3d_flatten_linear"
SAVE_NAME = "best_qtcn_3d_flatten_linear.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 2. Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

TRAIN_FILE = PROJECT_DIR / "datasets" / "ECG5000" / "ECG5000_train_split.txt"
VAL_FILE = PROJECT_DIR / "datasets" / "ECG5000" / "ECG5000_val_split.txt"
TEST_FILE = PROJECT_DIR / "datasets" / "ECG5000" / "ECG5000_TEST.txt"


# =========================================================
# 3. Reproducibility
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Keep this False unless you really need strict reproducibility.
    # Some older FINN/Brevitas/PyTorch environments may not support all
    # deterministic kernels.
    torch.backends.cudnn.benchmark = False


# =========================================================
# 4. Model import
# =========================================================
def import_model_class():
    """
    Preferred:
        newnet_3d_flatten_linear.py

    Fallback:
        newnet.py

    This makes the script usable whether you keep the 3D model under its
    explicit filename or replace your original newnet.py with it.
    """
    try:
        module = importlib.import_module(MODEL_MODULE_NAME)
        print(f"[info] Imported ECG5000FullQuantTCN from {MODEL_MODULE_NAME}.py")
    except ModuleNotFoundError:
        module = importlib.import_module("newnet")
        print("[warn] Could not find newnet_3d_flatten_linear.py; imported from newnet.py instead.")

    return module.ECG5000FullQuantTCN


# =========================================================
# 5. Data loading
# =========================================================
def load_ecg_txt(path: Path):
    """
    Read ECG5000 txt:
        each row: label feat1 feat2 ... feat140

    Returns:
        x: torch.FloatTensor [N, 1, 140]
        y: torch.LongTensor  [N]

    This is different from the previous 2D-conv FINN-friendly version,
    which used [N, 1, 140, 1]. This script is for the 3D feature model.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = np.loadtxt(path, dtype=np.float32)

    if data.ndim != 2 or data.shape[1] != SEQ_LEN + 1:
        raise ValueError(
            f"Expected txt shape [N, {SEQ_LEN + 1}] with label + {SEQ_LEN} values, "
            f"got {data.shape} from {path}"
        )

    y_np = data[:, 0].astype(np.int64)
    x_np = data[:, 1:]

    # ECG5000 labels are usually 1..5. Convert to 0..4.
    if y_np.min() >= 1:
        y_np = y_np - 1

    if y_np.min() < 0 or y_np.max() >= NUM_CLASSES:
        raise ValueError(
            f"Labels must be in 0..{NUM_CLASSES - 1} after remapping, "
            f"got min={y_np.min()}, max={y_np.max()}"
        )

    # [N, 140] -> [N, 1, 140]
    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y_np, dtype=torch.long)

    return x, y


def build_loader(path: Path, batch_size: int, shuffle: bool):
    x, y = load_ecg_txt(path)
    ds = TensorDataset(x, y)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, x, y


# =========================================================
# 6. Class weights
# =========================================================
def compute_class_weights(y: torch.Tensor, num_classes: int):
    counts = torch.bincount(y, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (counts * num_classes)
    return weights


# =========================================================
# 7. QuantTensor / logits handling
# =========================================================
def qt_value(x):
    return x.value if hasattr(x, "value") else x


def logits_to_2d(logits):
    """
    Convert model output to [N, 5] for CrossEntropyLoss.

    Main expected output for this model:
        [N, 5]

    Extra cases are kept only for debugging if you accidentally import
    an older network version.
    """
    logits = qt_value(logits)

    if logits.dim() == 2:
        if logits.shape[1] != NUM_CLASSES:
            raise ValueError(f"Expected logits shape [N, {NUM_CLASSES}], got {tuple(logits.shape)}")
        return logits

    # Debug compatibility with the old 2D conv classifier version.
    if logits.dim() == 4:
        if logits.shape[1] == NUM_CLASSES and logits.shape[2] == 1 and logits.shape[3] == 1:
            return logits[:, :, 0, 0]
        if logits.shape[1] == NUM_CLASSES and logits.shape[3] == 1:
            return logits[:, :, -1, 0]
        if logits.shape[-1] == NUM_CLASSES:
            return logits[:, -1, 0, :]

    raise ValueError(f"Expected logits convertible to [N, {NUM_CLASSES}], got {tuple(logits.shape)}")


def print_model_output_shape(model: nn.Module, device: torch.device):
    """
    Sanity check before training.
    Desired:
        input  [1, 1, 140]
        output [1, 5]
    """
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, NUM_INPUTS, SEQ_LEN, dtype=torch.float32).to(device)
        out = model(dummy)
        out_val = qt_value(out)

        print("[info] Dummy input shape       :", tuple(dummy.shape))
        print("[info] Dummy raw output shape  :", tuple(out_val.shape))
        print("[info] Dummy logits_2d shape   :", tuple(logits_to_2d(out).shape))

        if tuple(out_val.shape) != (1, NUM_CLASSES):
            print(
                "[warn] For the 3D feature + Flatten + Linear model, expected output "
                f"shape [1, {NUM_CLASSES}], but got {tuple(out_val.shape)}. "
                "Check that you imported the correct newnet_3d_flatten_linear.py."
            )


# =========================================================
# 8. Evaluation
# =========================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device, num_classes: int = 5):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)

    pred_all = []
    label_all = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        logits_val = logits_to_2d(logits)

        loss = criterion(logits_val, y)

        total_loss += loss.item() * x.size(0)

        pred = logits_val.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_samples += x.size(0)

        pred_cpu = pred.detach().cpu()
        y_cpu = y.detach().cpu()

        pred_all.append(pred_cpu)
        label_all.append(y_cpu)

        for c in range(num_classes):
            mask = y_cpu == c
            class_total[c] += mask.sum().item()
            class_correct[c] += ((pred_cpu == y_cpu) & mask).sum().item()

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    per_class_acc = []
    for c in range(num_classes):
        if class_total[c] > 0:
            per_class_acc.append(class_correct[c].item() / class_total[c].item())
        else:
            per_class_acc.append(0.0)

    pred_all = torch.cat(pred_all, dim=0)
    label_all = torch.cat(label_all, dim=0)

    pred_counts = torch.bincount(pred_all, minlength=num_classes)
    label_counts = torch.bincount(label_all, minlength=num_classes)

    return {
        "loss": avg_loss,
        "acc": avg_acc,
        "per_class_acc": per_class_acc,
        "pred_counts": pred_counts.tolist(),
        "label_counts": label_counts.tolist(),
    }


# =========================================================
# 9. Model construction
# =========================================================
def build_model():
    ECG5000FullQuantTCN = import_model_class()

    model = ECG5000FullQuantTCN(
        num_classes=NUM_CLASSES,
        seq_len=SEQ_LEN,
        num_inputs=NUM_INPUTS,
        num_channels=NUM_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        causal=CAUSAL,
    )

    return model


# =========================================================
# 10. Checkpoint helpers
# =========================================================
def save_checkpoint(
    save_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer,
    best_val_acc: float,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "model_config": {
                "num_classes": NUM_CLASSES,
                "seq_len": SEQ_LEN,
                "num_inputs": NUM_INPUTS,
                "num_channels": NUM_CHANNELS,
                "kernel_size": KERNEL_SIZE,
                "dropout": DROPOUT,
                "causal": CAUSAL,
                "feature_layout": "[N, C, L]",
                "input_shape": "[N, 1, 140]",
                "classifier": "Flatten + QuantLinear",
                "expected_output_shape": "[N, 5]",
            },
        },
        save_path,
    )


# =========================================================
# 11. Main
# =========================================================
def main():
    set_seed(SEED)

    print("=" * 70)
    print("Device    :", DEVICE)
    print("Base dir  :", BASE_DIR)
    print("Train file:", TRAIN_FILE)
    print("Val file  :", VAL_FILE)
    print("Test file :", TEST_FILE)
    print("Input     : [N, 1, 140]")
    print("Output    : [N, 5]")
    print("=" * 70)

    train_loader, x_train, y_train = build_loader(TRAIN_FILE, BATCH_SIZE, shuffle=True)
    val_loader, _, _ = build_loader(VAL_FILE, BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples  : {len(val_loader.dataset)}")
    print("Train x shape:", tuple(x_train.shape))
    print("Train y shape:", tuple(y_train.shape))

    train_counts = torch.bincount(y_train, minlength=NUM_CLASSES)
    print("Train class counts:", train_counts.tolist())

    model = build_model().to(DEVICE)
    print_model_output_shape(model, DEVICE)

    if USE_CLASS_WEIGHT:
        class_weights = compute_class_weights(y_train, NUM_CLASSES).to(DEVICE)
        print("Class weights:", class_weights.detach().cpu().numpy())
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
    )

    best_val_acc = -1.0
    best_epoch = -1
    best_state_dict = None

    save_path = BASE_DIR / SAVE_NAME

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            logits_val = logits_to_2d(logits)

            loss = criterion(logits_val, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = logits_val.argmax(dim=1)
            running_correct += (pred == y).sum().item()
            running_total += x.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_metrics = evaluate(model, val_loader, criterion, DEVICE, NUM_CLASSES)
        scheduler.step(val_metrics["acc"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{NUM_EPOCHS}] "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc']:.4f}"
        )
        print(
            "Val per-class acc:",
            [f"class {i}: {acc:.4f}" for i, acc in enumerate(val_metrics["per_class_acc"])],
        )
        print("Val label counts:", val_metrics["label_counts"])
        print("Val pred counts :", val_metrics["pred_counts"])

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

            save_checkpoint(
                save_path=save_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_val_acc=best_val_acc,
            )
            print(f"[info] Saved best model to: {save_path}")

    print("=" * 70)
    print(f"Training finished. Best val acc = {best_val_acc:.4f} at epoch {best_epoch}")
    print("=" * 70)

    if TEST_FILE.exists():
        print("Found test file, evaluating on test set...")

        if best_state_dict is None:
            raise RuntimeError("best_state_dict is None. No best model was saved during training.")

        model.load_state_dict(best_state_dict, strict=False)

        test_loader, _, _ = build_loader(TEST_FILE, BATCH_SIZE, shuffle=False)
        test_metrics = evaluate(model, test_loader, criterion, DEVICE, NUM_CLASSES)

        print(f"Test Loss: {test_metrics['loss']:.4f} | Test Acc: {test_metrics['acc']:.4f}")
        print(
            "Test per-class acc:",
            [f"class {i}: {acc:.4f}" for i, acc in enumerate(test_metrics["per_class_acc"])],
        )
        print("Test label counts:", test_metrics["label_counts"])
        print("Test pred counts :", test_metrics["pred_counts"])
    else:
        print(f"[warn] Test file does not exist, skipped test evaluation: {TEST_FILE}")


if __name__ == "__main__":
    main()
