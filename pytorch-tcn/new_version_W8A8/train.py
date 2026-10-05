import copy
import inspect
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from newnet import ECG5000FullQuantTCN


# =========================================================
# 1. Hyperparameters
# =========================================================
NUM_CLASSES = 5
SEQ_LEN = 140
NUM_INPUTS = 1
NUM_CHANNELS = (16, 16, 16)
KERNEL_SIZE = 3
DROPOUT = 0.1
CAUSAL = True

BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100

USE_CLASS_WEIGHT = True
SAVE_NAME = "best_qtcn_model_last_step.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This training script assumes the model should output only the final
# classification logits for FINN deployment.
# Recommended model forward output shape:
#     [N, 5, 1, 1]
# Training converts this to [N, 5] before CrossEntropyLoss.
EXPECT_LAST_STEP_ONLY_OUTPUT = True


# =========================================================
# 2. Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

TRAIN_FILE = PROJECT_DIR / "datasets" / "ECG5000" / "ECG5000_train_split.txt"
VAL_FILE = PROJECT_DIR / "datasets" / "ECG5000" / "ECG5000_val_split.txt"
TEST_FILE = PROJECT_DIR / "datasets" / "ECG5000" / "ECG5000_TEST.txt"


# =========================================================
# 3. Data loading
# =========================================================
def load_ecg_txt(path):
    """
    Read ECG5000 txt:
        each row: label feat1 feat2 ... feat140

    Returns:
        x: torch.FloatTensor [N, 1, 140, 1]
        y: torch.LongTensor  [N]

    The model forward you showed expects [N, C, L, 1], so the time dimension
    is dimension 2 and W is dimension 3.
    """
    data = np.loadtxt(path, dtype=np.float32)

    y = data[:, 0].astype(np.int64)
    x = data[:, 1:]

    if x.shape[1] != SEQ_LEN:
        raise ValueError(f"Expected {SEQ_LEN} ECG values, got {x.shape[1]}")

    # ECG5000 labels are usually 1..5, convert to 0..4.
    y = y - 1

    # [N, 140] -> [N, 1, 140, 1]
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.long)

    return x, y


def build_loader(path, batch_size, shuffle):
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
# 4. Class weights
# =========================================================
def compute_class_weights(y, num_classes):
    counts = torch.bincount(y, minlength=num_classes).float()

    # Avoid division by zero if a class is missing in a small split.
    counts = torch.clamp(counts, min=1.0)

    weights = counts.sum() / (counts * num_classes)
    return weights


# =========================================================
# 5. QuantTensor / logits handling
# =========================================================
def qt_value(x):
    return x.value if hasattr(x, "value") else x


def logits_to_2d(logits):
    """
    Convert model output to [N, 5] for CrossEntropyLoss.

    Preferred output after modifying the model:
        [N, 5, 1, 1] -> [N, 5]

    Also supports the old all-time output during debugging:
        [N, 5, 140, 1] -> [N, 5] using the last time step
    """
    logits = qt_value(logits)

    if logits.dim() == 2:
        if logits.shape[1] != NUM_CLASSES:
            raise ValueError(f"Expected logits shape [N, {NUM_CLASSES}], got {tuple(logits.shape)}")
        return logits

    if logits.dim() == 4:
        # Preferred FINN-friendly last-step output: [N, 5, 1, 1]
        if logits.shape[1] == NUM_CLASSES and logits.shape[2] == 1 and logits.shape[3] == 1:
            return logits[:, :, 0, 0]

        # Old all-time classifier output: [N, 5, 140, 1]
        if logits.shape[1] == NUM_CLASSES and logits.shape[3] == 1:
            if EXPECT_LAST_STEP_ONLY_OUTPUT:
                print(
                    "[warn] Model returned all-time logits [N, 5, L, 1]. "
                    "Training will use the last time step, but FINN deployment "
                    "will still output all time steps unless the model forward is changed."
                )
            return logits[:, :, -1, 0]

        # Possible NHWC-like layout after some transformations: [N, L, 1, 5]
        if logits.shape[-1] == NUM_CLASSES:
            return logits[:, -1, 0, :]

    raise ValueError(f"Expected logits convertible to [N, C], got {tuple(logits.shape)}")


def print_model_output_shape(model, device):
    """
    Quick sanity check for the output shape before training.
    The desired shape for FINN deployment is [1, 5, 1, 1].
    """
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, NUM_INPUTS, SEQ_LEN, 1, dtype=torch.float32).to(device)
        out = model(dummy)
        out_val = qt_value(out)
        print("[info] Dummy output shape:", tuple(out_val.shape))
        print("[info] Dummy logits_to_2d shape:", tuple(logits_to_2d(out).shape))

        if EXPECT_LAST_STEP_ONLY_OUTPUT:
            if tuple(out_val.shape) not in [(1, NUM_CLASSES, 1, 1), (1, NUM_CLASSES)]:
                print(
                    "[warn] Expected last-step-only output [N, 5, 1, 1] or [N, 5], "
                    f"but got {tuple(out_val.shape)}. "
                    "For FINN deployment, check that newnet.forward returns x[:, :, -1:, :] "
                    "after the classifier."
                )


# =========================================================
# 6. Evaluation
# =========================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=5):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)

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

        for c in range(num_classes):
            mask = y == c
            class_total[c] += mask.sum().item()
            class_correct[c] += ((pred == y) & mask).sum().item()

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    per_class_acc = []
    for c in range(num_classes):
        if class_total[c] > 0:
            per_class_acc.append(class_correct[c].item() / class_total[c].item())
        else:
            per_class_acc.append(0.0)

    return avg_loss, avg_acc, per_class_acc


# =========================================================
# 7. Model construction
# =========================================================
def build_model():
    """
    Construct ECG5000FullQuantTCN robustly.

    If your newnet.ECG5000FullQuantTCN supports export_all_time_logits,
    this passes export_all_time_logits=False. If it does not support that
    argument, the constructor falls back to the original signature.

    The most important change is still inside newnet.forward:
        x = self.classifier(x)
        x = qt_value(x)
        x = x[:, :, -1:, :]
        return x
    """
    common_kwargs = dict(
        num_classes=NUM_CLASSES,
        seq_len=SEQ_LEN,
        num_inputs=NUM_INPUTS,
        num_channels=NUM_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        causal=CAUSAL,
    )

    try:
        sig = inspect.signature(ECG5000FullQuantTCN)
        if "export_all_time_logits" in sig.parameters:
            common_kwargs["export_all_time_logits"] = False
    except Exception:
        pass

    return ECG5000FullQuantTCN(**common_kwargs)


# =========================================================
# 8. Main
# =========================================================
def main():
    print("=" * 60)
    print("Device:", DEVICE)
    print("Train file:", TRAIN_FILE)
    print("Val file  :", VAL_FILE)
    print("Test file :", TEST_FILE)
    print("=" * 60)

    train_loader, x_train, y_train = build_loader(TRAIN_FILE, BATCH_SIZE, shuffle=True)
    val_loader, _, _ = build_loader(VAL_FILE, BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples  : {len(val_loader.dataset)}")

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

    best_val_acc = 0.0
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

            optimizer.zero_grad()

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

        val_loss, val_acc, val_per_class_acc = evaluate(
            model, val_loader, criterion, DEVICE, NUM_CLASSES
        )

        scheduler.step(val_acc)

        print(
            f"Epoch [{epoch:03d}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        print(
            "Val per-class acc:",
            [f"class {i}: {acc:.4f}" for i, acc in enumerate(val_per_class_acc)],
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state_dict,
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
                        "export_all_time_logits": False,
                        "expected_output_shape": "[N, 5, 1, 1]",
                    },
                },
                save_path,
            )
            print(f"Saved best model to: {save_path}")

    print("=" * 60)
    print(f"Training finished. Best val acc = {best_val_acc:.4f} at epoch {best_epoch}")
    print("=" * 60)

    # Use the in-memory best weights for testing to avoid possible Brevitas
    # buffer mismatch after reloading.
    if TEST_FILE.exists():
        print("Found test file, evaluating on test set...")

        if best_state_dict is None:
            raise RuntimeError("best_state_dict is None, no best model was saved during training.")

        model.load_state_dict(best_state_dict, strict=False)

        test_loader, _, _ = build_loader(TEST_FILE, BATCH_SIZE, shuffle=False)
        test_loss, test_acc, test_per_class_acc = evaluate(
            model, test_loader, criterion, DEVICE, NUM_CLASSES
        )

        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        print(
            "Test per-class acc:",
            [f"class {i}: {acc:.4f}" for i, acc in enumerate(test_per_class_acc)],
        )


if __name__ == "__main__":
    main()
