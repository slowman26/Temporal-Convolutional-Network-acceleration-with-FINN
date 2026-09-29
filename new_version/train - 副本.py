import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# 1. 改这里：导入你的模型
# =========================================================
# 你把这里改成你实际的模型路径
# 例如：
# from solid_net.ECG5000_QTCN_solid import ECG5000FullQuantTCN

#from solid_net.ECG5000_QTCN_solid import ECG5000FullQuantTCN
from newnet import ECG5000FullQuantTCN


# =========================================================
# 2. 超参数
# =========================================================
NUM_CLASSES = 5
SEQ_LEN = 140
NUM_INPUTS = 1
NUM_CHANNELS = (16, 16, 32)
KERNEL_SIZE = 3
DROPOUT = 0.1
CAUSAL = False

BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20

USE_CLASS_WEIGHT = True
SAVE_NAME = "best_qtcn_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 3. 路径
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

TRAIN_FILE = PROJECT_DIR / "datasets" / "ECG5000" / "ECG5000_train_split.txt"
VAL_FILE = PROJECT_DIR / "datasets" / "ECG5000" / "ECG5000_val_split.txt"

# 可选：如果你后面想顺便测 test
TEST_FILE = PROJECT_DIR / "datasets" / "ECG5000" / "ECG5000_TEST.txt"


# =========================================================
# 4. 数据读取
# =========================================================
def load_ecg_txt(path):
    """
    读取 txt:
        每行格式: label feat1 feat2 ... feat140
    返回:
        x: torch.FloatTensor [N, 1, 140]
        y: torch.LongTensor  [N]
    """
    data = np.loadtxt(path, dtype=np.float32)

    y = data[:, 0].astype(np.int64)
    x = data[:, 1:]

    # ECG5000 原始标签通常是 1~5，转成 0~4
    y = y - 1

    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1).unsqueeze(2)  # [N, 1, 140]
    print("x=",x.shape)
    y = torch.tensor(y, dtype=torch.long)
    print("y=",y.shape)

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
# 5. 类别权重（可选，但我建议你开着）
# ECG5000 类别很不均衡，不加权的话小类很容易学不好
# =========================================================
def compute_class_weights(y, num_classes):
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = counts.sum() / (counts * num_classes)
    return weights

def qt_value(x):
    return x.value if hasattr(x, "value") else x

# =========================================================
# 6. 评估函数
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

        #new
        logits_val = qt_value(logits)

        loss = criterion(logits_val, y)

        total_loss += loss.item() * x.size(0)

        pred = logits_val.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_samples += x.size(0)

        for c in range(num_classes):
            mask = (y == c)
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
# 7. 训练主函数
# =========================================================
def main():
    print("=" * 60)
    print("Device:", DEVICE)
    print("Train file:", TRAIN_FILE)
    print("Val file  :", VAL_FILE)
    print("=" * 60)

    train_loader, x_train, y_train = build_loader(TRAIN_FILE, BATCH_SIZE, shuffle=True)
    val_loader, _, _ = build_loader(VAL_FILE, BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples  : {len(val_loader.dataset)}")

    # 类别统计
    train_counts = torch.bincount(y_train, minlength=NUM_CLASSES)
    print("Train class counts:", train_counts.tolist())

    # 构建模型
    model = ECG5000FullQuantTCN(
        num_classes=NUM_CLASSES,
        seq_len=SEQ_LEN,
        num_inputs=NUM_INPUTS,
        num_channels=NUM_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        causal=CAUSAL,
    ).to(DEVICE)

    # Loss
    if USE_CLASS_WEIGHT:
        class_weights = compute_class_weights(y_train, NUM_CLASSES).to(DEVICE)
        print("Class weights:", class_weights.detach().cpu().numpy())
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Scheduler（可选）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
    )

    best_val_acc = 0.0
    best_epoch = -1

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

            #new
            logits_val = qt_value(logits)

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
            [f"class {i}: {acc:.4f}" for i, acc in enumerate(val_per_class_acc)]
        )

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
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
                    },
                },
                save_path,
            )
            print(f"Saved best model to: {save_path}")

    print("=" * 60)
    print(f"Training finished. Best val acc = {best_val_acc:.4f} at epoch {best_epoch}")
    print("=" * 60)

    # =====================================================
    # 8. 可选：训练结束后顺手在 test 上评估
    # =====================================================
    if TEST_FILE.exists():
        print("Found test file, evaluating on test set...")

        checkpoint = torch.load(save_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_loader, _, _ = build_loader(TEST_FILE, BATCH_SIZE, shuffle=False)
        test_loss, test_acc, test_per_class_acc = evaluate(
            model, test_loader, criterion, DEVICE, NUM_CLASSES
        )

        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        print(
            "Test per-class acc:",
            [f"class {i}: {acc:.4f}" for i, acc in enumerate(test_per_class_acc)]
        )


if __name__ == "__main__":
    main()