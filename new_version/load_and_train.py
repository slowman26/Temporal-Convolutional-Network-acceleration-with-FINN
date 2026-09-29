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

# Fine-tuning should usually use smaller LR than training from scratch.
LR = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 80

USE_CLASS_WEIGHT = True
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")


EXPECT_LAST_STEP_ONLY_OUTPUT = True


# =========================================================
# 2. Quantization target
# =========================================================
# Example:
#   W4A4: weight_bits=4, act_bits=4, input_bits=4
#   W2A4: weight_bits=2, act_bits=4, input_bits=4
#   W1A4: weight_bits=1, act_bits=4, input_bits=4
TARGET_WEIGHT_BITS = 4
TARGET_ACT_BITS = 1
TARGET_INPUT_BITS = 4


# =========================================================
# 3. Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

TRAIN_FILE = PROJECT_DIR / "datasets" / "ECG5000" / "ECG5000_train_split.txt"
VAL_FILE = PROJECT_DIR / "datasets" / "ECG5000" / "ECG5000_val_split.txt"
TEST_FILE = PROJECT_DIR / "datasets" / "ECG5000" / "ECG5000_TEST.txt"

# Old checkpoint, for example INT8-trained model.
LOAD_CKPT = BASE_DIR / "best_qtcn_model_W4A2_finetune.pth"

# New fine-tuned checkpoint.
SAVE_NAME = f"best_qtcn_model_W{TARGET_WEIGHT_BITS}A{TARGET_ACT_BITS}_finetune.pth"
SAVE_PATH = BASE_DIR / SAVE_NAME


# =========================================================
# 4. Data loading
# =========================================================
def load_ecg_txt(path):
    """
    ECG5000 txt format:
        label feat1 feat2 ... feat140

    Returns:
        x: [N, 1, 140, 1]
        y: [N]
    """
    data = np.loadtxt(path, dtype=np.float32)

    y = data[:, 0].astype(np.int64)
    x = data[:, 1:]

    if x.shape[1] != SEQ_LEN:
        raise ValueError(f"Expected {SEQ_LEN} ECG values, got {x.shape[1]}")

    # ECG5000 labels are usually 1..5. Convert to 0..4.
    y = y - 1

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
        pin_memory=False,
    )

    return loader, x, y


# =========================================================
# 5. Class weights
# =========================================================
def compute_class_weights(y, num_classes):
    counts = torch.bincount(y, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (counts * num_classes)
    return weights


# =========================================================
# 6. QuantTensor / logits handling
# =========================================================
def qt_value(x):
    return x.value if hasattr(x, "value") else x


def logits_to_2d(logits):
    """
    Convert model output to [N, 5].
    Supports:
        [N, 5]
        [N, 5, 1, 1]
        [N, 5, L, 1]
        [N, L, 1, 5]
    """
    logits = qt_value(logits)

    if logits.dim() == 2:
        if logits.shape[1] != NUM_CLASSES:
            raise ValueError(f"Expected logits shape [N, {NUM_CLASSES}], got {tuple(logits.shape)}")
        return logits

    if logits.dim() == 4:
        # Preferred FINN-friendly output: [N, 5, 1, 1]
        if logits.shape[1] == NUM_CLASSES and logits.shape[2] == 1 and logits.shape[3] == 1:
            return logits[:, :, 0, 0]

        # All-time output: [N, 5, L, 1]
        if logits.shape[1] == NUM_CLASSES and logits.shape[3] == 1:
            if EXPECT_LAST_STEP_ONLY_OUTPUT:
                print(
                    "[warn] Model returned all-time logits [N, 5, L, 1]. "
                    "Training uses the last time step."
                )
            return logits[:, :, -1, 0]

        # NHWC-like output: [N, L, 1, 5]
        if logits.shape[-1] == NUM_CLASSES:
            return logits[:, -1, 0, :]

    raise ValueError(f"Expected logits convertible to [N, C], got {tuple(logits.shape)}")


def print_model_output_shape(model, device):
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
                    f"but got {tuple(out_val.shape)}."
                )


# =========================================================
# 7. Build new quantized model
# =========================================================
def build_model():
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

        if "weight_bits" in sig.parameters:
            common_kwargs["weight_bits"] = TARGET_WEIGHT_BITS

        if "act_bits" in sig.parameters:
            common_kwargs["act_bits"] = TARGET_ACT_BITS

        if "input_bits" in sig.parameters:
            common_kwargs["input_bits"] = TARGET_INPUT_BITS

    except Exception as e:
        print("[warn] Could not inspect model signature:", repr(e))

    model = ECG5000FullQuantTCN(**common_kwargs)
    return model


# =========================================================
# 8. Load old checkpoint into new model
# =========================================================
def strip_module_prefix(state_dict):
    """
    Remove 'module.' prefix if the model was trained with DataParallel.
    """
    new_state = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v

    return new_state


def extract_model_state_dict(ckpt):
    """
    Supports both:
        torch.save(model.state_dict())
    and:
        torch.save({"model_state_dict": model.state_dict(), ...})
    """
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]

    return ckpt


def load_pretrained_for_finetune(model, ckpt_path, device):
    """
    Load old checkpoint into the new quantized model.

    Only parameters with matching names and matching shapes are loaded.
    This avoids errors from changed quantizer buffers.
    """
    if not ckpt_path.exists():
        print(f"[warn] Checkpoint not found: {ckpt_path}")
        print("[warn] Training will start from scratch.")
        return model

    print("=" * 60)
    print("[info] Loading pretrained checkpoint:")
    print("       ", ckpt_path)
    print("=" * 60)

    ckpt = torch.load(ckpt_path, map_location=device)
    old_state = extract_model_state_dict(ckpt)
    old_state = strip_module_prefix(old_state)

    model_state = model.state_dict()

    loaded_state = {}
    skipped_keys = []

    for k, v in old_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            loaded_state[k] = v
        else:
            if k in model_state:
                skipped_keys.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            else:
                skipped_keys.append((k, tuple(v.shape), None))

    missing, unexpected = model.load_state_dict(loaded_state, strict=False)

    print(f"[info] Loaded tensors : {len(loaded_state)}")
    print(f"[info] Missing keys    : {len(missing)}")
    print(f"[info] Unexpected keys : {len(unexpected)}")
    print(f"[info] Skipped keys    : {len(skipped_keys)}")

    if len(skipped_keys) > 0:
        print("[info] First skipped keys:")
        for item in skipped_keys[:20]:
            print("   ", item)

    print("=" * 60)

    return model


# =========================================================
# 9. Evaluation
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
# 10. Main
# =========================================================
def main():
    print("=" * 60)
    print("Device:", DEVICE)
    print("Train file:", TRAIN_FILE)
    print("Val file  :", VAL_FILE)
    print("Test file :", TEST_FILE)
    print("Load ckpt :", LOAD_CKPT)
    print("Save path :", SAVE_PATH)
    print("Target quant:")
    print(f"  weight_bits = {TARGET_WEIGHT_BITS}")
    print(f"  act_bits    = {TARGET_ACT_BITS}")
    print(f"  input_bits  = {TARGET_INPUT_BITS}")
    print("=" * 60)

    train_loader, x_train, y_train = build_loader(TRAIN_FILE, BATCH_SIZE, shuffle=True)
    val_loader, _, _ = build_loader(VAL_FILE, BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples  : {len(val_loader.dataset)}")

    train_counts = torch.bincount(y_train, minlength=NUM_CLASSES)
    print("Train class counts:", train_counts.tolist())

    model = build_model().to(DEVICE)

    # Load old INT8 / previous checkpoint into the new lower-bit model.
    model = load_pretrained_for_finetune(model, LOAD_CKPT, DEVICE)
    model = model.to(DEVICE)

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

    # Evaluate once before fine-tuning.
    val_loss, val_acc, val_per_class_acc = evaluate(
        model, val_loader, criterion, DEVICE, NUM_CLASSES
    )

    print("=" * 60)
    print("[before fine-tune]")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    print(
        "Val per-class acc:",
        [f"class {i}: {acc:.4f}" for i, acc in enumerate(val_per_class_acc)],
    )
    print("=" * 60)

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

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{NUM_EPOCHS}] "
            f"LR: {current_lr:.2e} | "
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
                        "weight_bits": TARGET_WEIGHT_BITS,
                        "act_bits": TARGET_ACT_BITS,
                        "input_bits": TARGET_INPUT_BITS,
                        "export_all_time_logits": False,
                        "expected_output_shape": "[N, 5, 1, 1]",
                        "loaded_from": str(LOAD_CKPT),
                    },
                },
                SAVE_PATH,
            )

            print(f"[saved] Best fine-tuned model saved to: {SAVE_PATH}")

    print("=" * 60)
    print(f"Fine-tuning finished.")
    print(f"Best val acc = {best_val_acc:.4f} at epoch {best_epoch}")
    print("=" * 60)

    if TEST_FILE.exists():
        print("Found test file, evaluating on test set...")

        if best_state_dict is None:
            raise RuntimeError("best_state_dict is None. No best model was saved.")

        model.load_state_dict(best_state_dict, strict=False)

        test_loader, _, _ = build_loader(TEST_FILE, BATCH_SIZE, shuffle=False)

        test_loss, test_acc, test_per_class_acc = evaluate(
            model, test_loader, criterion, DEVICE, NUM_CLASSES
        )

        print("=" * 60)
        print("[test result]")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        print(
            "Test per-class acc:",
            [f"class {i}: {acc:.4f}" for i, acc in enumerate(test_per_class_acc)],
        )
        print("=" * 60)


if __name__ == "__main__":
    main()