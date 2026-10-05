from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pytorch_tcn.tcn import TCN


# =========================
# 1. 基本配置
# =========================
SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4

VAL_RATIO = 0.2

NUM_CLASSES = 5
INPUT_CHANNELS = 1
SEQ_LEN = 140

NUM_CHANNELS = [16, 16, 32]
KERNEL_SIZE = 3
DROPOUT = 0.1

SAVE_NAME = "best_fp_tcn_model.pth"


# =========================
# 2. 固定随机种子
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 3. 数据读取
# =========================
def load_ecg5000_txt(file_path):
    """
    ECG5000 txt 格式默认每行:
    label, x1, x2, ..., x140
    或者空格分隔:
    label x1 x2 ... x140
    """
    data = np.loadtxt(file_path, dtype=np.float32)

    y = data[:, 0].astype(np.int64)
    x = data[:, 1:].astype(np.float32)

    # ECG5000 常见标签是 1~5，转成 0~4
    if y.min() == 1:
        y = y - 1

    # [N, 140] -> [N, 1, 140]
    x = x[:, np.newaxis, :]

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return x, y


def stratified_split(x, y, val_ratio=0.2, seed=42):
    """
    不依赖 sklearn 的分层划分
    """
    rng = np.random.default_rng(seed)

    x_np = x.numpy()
    y_np = y.numpy()

    train_indices = []
    val_indices = []

    classes = np.unique(y_np)
    for c in classes:
        cls_idx = np.where(y_np == c)[0]
        rng.shuffle(cls_idx)

        n_val = max(1, int(len(cls_idx) * val_ratio))
        val_idx_c = cls_idx[:n_val]
        train_idx_c = cls_idx[n_val:]

        train_indices.extend(train_idx_c.tolist())
        val_indices.extend(val_idx_c.tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    x_train = torch.tensor(x_np[train_indices], dtype=torch.float32)
    y_train = torch.tensor(y_np[train_indices], dtype=torch.long)

    x_val = torch.tensor(x_np[val_indices], dtype=torch.float32)
    y_val = torch.tensor(y_np[val_indices], dtype=torch.long)

    return x_train, y_train, x_val, y_val



class ECG5000FloatTCNClassifier(nn.Module):
    def __init__(
        self,
        num_classes=5,
        seq_len=140,
        num_inputs=1,
        num_channels=(16, 16, 32),
        kernel_size=3,
        dropout=0.1,
        causal=False,
        use_skip_connections=False,
        input_shape='NCL',
    ):
        super().__init__()

        self.backbone = TCN(
            num_inputs=num_inputs,
            num_channels=list(num_channels),
            kernel_size=kernel_size,
            dropout=dropout,
            causal=causal,
            use_skip_connections=use_skip_connections,
            input_shape=input_shape,
        )

        self.classifier = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # x: [N, C, L]
        x = self.backbone(x)      # 通常输出 [N, C_out, L]
        x = x[:, :, -1]           # 取最后一个时间步
        x = self.classifier(x)
        return x


# =========================
# 5. 评估函数
# =========================
def evaluate(model, loader, device, num_classes=5):
    model.eval()

    total = 0
    correct = 0

    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            preds = torch.argmax(logits, dim=1)

            total += yb.size(0)
            correct += (preds == yb).sum().item()

            for c in range(num_classes):
                mask = (yb == c)
                class_total[c] += mask.sum().item()
                class_correct[c] += ((preds == yb) & mask).sum().item()

    acc = correct / total

    per_class_acc = []
    for c in range(num_classes):
        if class_total[c] == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append(class_correct[c] / class_total[c])

    return acc, per_class_acc


# =========================
# 6. 主训练流程
# =========================
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = Path(__file__).resolve().parents[1]
    train_file = root / "datasets" / "ECG5000" / "ECG5000_TRAIN.txt"
    test_file = root / "datasets" / "ECG5000" / "ECG5000_TEST.txt"
    save_path = root / SAVE_NAME

    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    # 读取数据
    x_train_full, y_train_full = load_ecg5000_txt(train_file)
    x_test, y_test = load_ecg5000_txt(test_file)

    # train / val 划分
    x_train, y_train, x_val, y_val = stratified_split(
        x_train_full, y_train_full, val_ratio=VAL_RATIO, seed=SEED
    )

    print("Train:", x_train.shape, y_train.shape)
    print("Val:  ", x_val.shape, y_val.shape)
    print("Test: ", x_test.shape, y_test.shape)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = ECG5000FloatTCNClassifier(
        num_inputs=INPUT_CHANNELS,
        num_channels=NUM_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * yb.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == yb).sum().item()
            running_total += yb.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_acc, val_per_class = evaluate(model, val_loader, device, NUM_CLASSES)

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "config": {
                        "input_channels": INPUT_CHANNELS,
                        "num_channels": NUM_CHANNELS,
                        "kernel_size": KERNEL_SIZE,
                        "dropout": DROPOUT,
                        "num_classes": NUM_CLASSES,
                    }
                },
                save_path
            )
            print(f"**** Best model saved to {save_path.name}, Val Acc: {best_val_acc:.4f}")

    print(f"\nTraining finished. Best Val Acc: {best_val_acc:.4f}")

    # 加载最佳模型并测试
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_acc, test_per_class = evaluate(model, test_loader, device, NUM_CLASSES)

    print(f"\nTest Accuracy: {test_acc:.4f}")
    print("\nPer-class results:")
    for i, acc in enumerate(test_per_class):
        print(f"Class {i}: {acc:.4f}")


if __name__ == "__main__":
    main()