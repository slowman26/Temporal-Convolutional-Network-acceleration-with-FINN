import os
import torch
import torch.nn as nn
import numpy as np
from quant_tcn.ECG5000_QTCN import ECG5000QTCNClassifier
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

base_dir = Path(__file__).resolve().parent          # train.py 所在目录
project_dir = base_dir.parent                       # pytorch-tcn 根目录

train_file = project_dir / "datasets" / "ECG5000" / "ECG5000_train_split.txt"
val_file = project_dir / "datasets" / "ECG5000" / "ECG5000_val_split.txt"

NUM_CLASSES = 5
SEQ_LEN = 140
NUM_INPUTS = 1
NUM_CHANNELS = (16, 16, 32)
KERNEL_SIZE = 3
DROPOUT = 0.1
CAUSAL = False
USE_SKIP_CONNECTIONS = False
INPUT_SHAPE = "NCL"

BATCH_SIZE = 64
LR = 1e-3
NUM_EPOCHS = 100


def load_ecg_txt(path):
    data = np.loadtxt(path)              # [N, 141]
    y = data[:, 0].astype(np.int64)      # label
    X = data[:, 1:].astype(np.float32)   # 140 points

    y = y - 1                            # 1~5 -> 0~4
    X = X[:, np.newaxis, :]              # [N, 1, 140]

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    return X, y

X_train, y_train = load_ecg_txt(train_file)
X_test, y_test = load_ecg_txt(val_file)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ECG5000QTCNClassifier(
    num_classes=NUM_CLASSES,
    seq_len=SEQ_LEN,
    num_inputs=NUM_INPUTS,
    num_channels=NUM_CHANNELS,
    kernel_size=KERNEL_SIZE,
    dropout=DROPOUT,
    causal=CAUSAL,
    use_skip_connections=USE_SKIP_CONNECTIONS,
    input_shape=INPUT_SHAPE,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 100
best_val_acc = 0.0
save_path = "best_qtcn_model.pth"

train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

for epoch in range(num_epochs):
    # =========================
    # 1. train
    # =========================
    model.train()
    running_loss = 0.0
    total_train = 0
    correct_train = 0

    for xb, yb in train_loader:
        xb = xb.to(device).float()   # [N, 1, 140]
        yb = yb.to(device).long()    # [N]

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        batch_size = xb.size(0)
        running_loss += loss.item() * batch_size
        total_train += batch_size

        pred = torch.argmax(logits, dim=1)
        correct_train += (pred == yb).sum().item()

    epoch_train_loss = running_loss / total_train
    epoch_train_acc = correct_train / total_train

    train_loss_history.append(epoch_train_loss)
    train_acc_history.append(epoch_train_acc)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {epoch_train_loss:.4f} "
        f"Train Acc: {epoch_train_acc:.4f}"
    )

    # =========================
    # 2. validate every 5 epochs
    # =========================
    if (epoch + 1) % 5 == 0:
        model.eval()
        val_loss = 0.0
        total_val = 0
        correct_val = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).float()
                yb = yb.to(device).long()

                logits = model(xb)
                loss = criterion(logits, yb)

                batch_size = xb.size(0)
                val_loss += loss.item() * batch_size
                total_val += batch_size

                pred = torch.argmax(logits, dim=1)
                correct_val += (pred == yb).sum().item()

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)

        print(
            f"---- Validation @ Epoch {epoch+1}: "
            f"Val Loss: {epoch_val_loss:.4f} "
            f"Val Acc: {epoch_val_acc:.4f}"
        )

        # 保存最优权重
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "config": {
                        "num_classes": NUM_CLASSES,
                        "seq_len": SEQ_LEN,
                        "num_inputs": NUM_INPUTS,
                        "num_channels": NUM_CHANNELS,
                        "kernel_size": KERNEL_SIZE,
                        "dropout": DROPOUT,
                        "causal": CAUSAL,
                        "use_skip_connections": USE_SKIP_CONNECTIONS,
                        "input_shape": INPUT_SHAPE,
                    },
                    "train_info": {
                        "num_epochs": NUM_EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LR,
                        "train_file": str(train_file),
                        "val_file": str(val_file),
                    }
                },
                save_path
            )
            print(f"**** Best model saved to {save_path}, Val Acc: {best_val_acc:.4f}")

print(f"Training finished. Best Val Acc: {best_val_acc:.4f}")