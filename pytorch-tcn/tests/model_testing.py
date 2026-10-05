from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# ===== 把项目根目录加入搜索路径 =====
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_tcn.ECG5000_QTCN import ECG5000QTCNClassifier


def load_ecg5000_txt(file_path):
    """
    读取 ECG5000 的 txt 文件
    假设每一行格式为:
    label feature1 feature2 ... feature140
    """
    data = np.loadtxt(file_path, dtype=np.float32)

    y = data[:, 0].astype(np.int64)
    X = data[:, 1:].astype(np.float32)

    # ECG5000 原始标签通常是 1~5，这里转成 0~4
    unique_labels = np.unique(y)
    if unique_labels.min() == 1:
        y = y - 1

    # 变成 [N, C, L] = [N, 1, 140]
    X = X[:, np.newaxis, :]

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return X, y


def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == yb).sum().item()
            total += yb.size(0)

            all_preds.append(preds.cpu())
            all_targets.append(yb.cpu())

    acc = correct / total
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return acc, all_preds, all_targets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== 路径 =====
    test_file = ROOT / "datasets" / "ECG5000" / "ECG5000_TEST.txt"
    ckpt_file = ROOT / "best_qtcn_model.pth"

    print(f"Test file: {test_file}")
    print(f"Checkpoint: {ckpt_file}")

    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    # ===== 读取测试集 =====
    X_test, y_test = load_ecg5000_txt(test_file)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=64,
        shuffle=False
    )

    # ===== 构建模型 =====
    # 如果你的 ECG5000QTCNClassifier 训练时传了参数，
    # 这里必须和训练时保持一致
    model = ECG5000QTCNClassifier()
    model = model.to(device)

    # ===== 加载权重 =====
    checkpoint = torch.load(ckpt_file, map_location=device)

    # 兼容两种保存方式：
    # 1) torch.save(model.state_dict(), path)
    # 2) torch.save({"model_state_dict": ...}, path)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print("Model weights loaded successfully.")

    # ===== 测试 =====
    test_acc, all_preds, all_targets = evaluate(model, test_loader, device)

    print(f"\nTest Accuracy: {test_acc:.4f}")

    # ===== 简单统计每类准确数，可选 =====
    num_classes = len(torch.unique(all_targets))
    print("\nPer-class results:")
    for c in range(num_classes):
        mask = (all_targets == c)
        class_total = mask.sum().item()
        class_correct = ((all_preds == all_targets) & mask).sum().item()
        class_acc = class_correct / class_total if class_total > 0 else 0.0
        print(f"Class {c}: {class_correct}/{class_total} = {class_acc:.4f}")


if __name__ == "__main__":
    main()