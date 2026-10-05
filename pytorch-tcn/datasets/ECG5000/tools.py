import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# =========================
# 路径设置
# =========================

base_dir = Path(__file__).resolve().parent
input_file = base_dir / "ECG5000_TRAIN.txt"

train_out_file = base_dir / "ECG5000_train_split.txt"
val_out_file = base_dir / "ECG5000_val_split.txt"

# 验证集比例
val_ratio = 0.2

# 随机种子，保证可复现
random_seed = 42

# =========================
# 读取原始训练集
# =========================
data = np.loadtxt(input_file)   # shape: [N, 141]

# 第一列是标签，后面是特征
y = data[:, 0]
X = data[:, 1:]

print("Original data shape:", data.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)

# 看一下类别分布
unique_labels, counts = np.unique(y, return_counts=True)
print("Original label distribution:")
for label, count in zip(unique_labels, counts):
    print(f"Class {int(label)}: {count}")

# =========================
# 分层划分 train / val
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=val_ratio,
    random_state=random_seed,
    stratify=y
)

print("\nAfter split:")
print("Train X shape:", X_train.shape)
print("Val X shape:", X_val.shape)
print("Train y shape:", y_train.shape)
print("Val y shape:", y_val.shape)

# 检查划分后的类别分布
print("\nTrain label distribution:")
unique_labels, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Class {int(label)}: {count}")

print("\nVal label distribution:")
unique_labels, counts = np.unique(y_val, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Class {int(label)}: {count}")

# =========================
# 拼回原格式：[label, features...]
# =========================
train_data = np.column_stack((y_train, X_train))
val_data = np.column_stack((y_val, X_val))

# =========================
# 保存
# =========================
np.savetxt(train_out_file, train_data, fmt="%.7e")
np.savetxt(val_out_file, val_data, fmt="%.7e")

print(f"\nSaved train split to: {train_out_file}")
print(f"Saved val split to: {val_out_file}")