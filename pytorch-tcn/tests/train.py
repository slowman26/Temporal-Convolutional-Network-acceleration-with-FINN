import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

#from solid_net.ECG5000_QTCN_solid import ECG5000QTCNClassifier
from solid_net.model_factory import build_export_model,build_train_model

# =========================
# paths
# =========================
base_dir = Path(__file__).resolve().parent          # tests/
project_dir = base_dir.parent                       # pytorch-tcn/

train_file = project_dir / "datasets" / "ECG5000" / "ECG5000_train_split.txt"
val_file = project_dir / "datasets" / "ECG5000" / "ECG5000_val_split.txt"
save_path = base_dir / "best_qtcn_model.pth"


# =========================
# hyperparameters
# =========================
NUM_CLASSES = 5
SEQ_LEN = 140
NUM_INPUTS = 1
NUM_CHANNELS = (16, 16, 16)
KERNEL_SIZE = 3
DROPOUT = 0.1
CAUSAL = True
USE_SKIP_CONNECTIONS = False
INPUT_SHAPE = "NCL"

BATCH_SIZE = 64
LR = 1e-3
NUM_EPOCHS = 20
SEED = 42


# =========================
# utils
# =========================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_ecg_txt(path):
    """
    txt format:
    each row = [label, x1, x2, ..., x140]
    label range: 1~5
    """
    data = np.loadtxt(path, dtype=np.float32)   # [N, 141]

    y = data[:, 0].astype(np.int64)             # [N]
    X = data[:, 1:]                             # [N, 140]

    y = y - 1                                   # 1~5 -> 0~4
    X = X[:, np.newaxis, :]                     # [N, 1, 140]

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    return X, y


def forward_model(model, xb):
    """
    统一处理不同 forward 返回格式：
    1. model(x) -> logits
    2. model(x) -> (logits, ...)
    3. model(x) -> {"logits": ...}
    """
    out = model(xb)

    if isinstance(out, dict):
        if "logits" in out:
            logits = out["logits"]
        else:
            raise ValueError(
                f"Model returned dict, but no 'logits' key found. Keys: {list(out.keys())}"
            )
    elif isinstance(out, (tuple, list)):
        if len(out) == 0:
            raise ValueError("Model returned an empty tuple/list.")
        logits = out[0]
    else:
        logits = out

    if not torch.is_tensor(logits):
        raise TypeError(f"Model output logits must be a tensor, but got {type(logits)}")

    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits shape [B, num_classes], but got {tuple(logits.shape)}"
        )

    return logits


def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = forward_model(model, xb)
            loss = criterion(logits, yb)

            batch_size = xb.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            pred = torch.argmax(logits, dim=1)
            correct += (pred == yb).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# =========================
# main
# =========================
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    X_train, y_train = load_ecg_txt(train_file)
    X_val, y_val = load_ecg_txt(val_file)

    print("Train:", X_train.shape, y_train.shape)
    print("Val  :", X_val.shape, y_val.shape)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    # model
    model = build_train_model(
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

    # quick sanity check
    with torch.no_grad():
        xb_test = X_train[:2].to(device)
        logits_test = forward_model(model, xb_test)
        print("Sanity check - input :", tuple(xb_test.shape))
        print("Sanity check - logits:", tuple(logits_test.shape))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(NUM_EPOCHS):
        model.train()

        running_loss = 0.0
        total_train = 0
        correct_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = forward_model(model, xb)
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

        epoch_val_loss, epoch_val_acc = evaluate(
            model, val_loader, criterion, device
        )

        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)

        print(
            f"Epoch [{epoch + 1:03d}/{NUM_EPOCHS}] | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.4f}"
        )

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
                    },
                    "history": {
                        "train_loss": train_loss_history,
                        "train_acc": train_acc_history,
                        "val_loss": val_loss_history,
                        "val_acc": val_acc_history,
                    },
                },
                save_path,
            )
            print(f"**** Best model saved to: {save_path}")
            print(f"**** Best Val Acc: {best_val_acc:.4f}")

    print(f"\nTraining finished. Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()