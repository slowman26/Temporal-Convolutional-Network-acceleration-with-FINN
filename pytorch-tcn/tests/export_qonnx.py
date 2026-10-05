from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_tcn.ECG5000_QTCN import ECG5000QTCNClassifier
from brevitas.export import export_qonnx

from qonnx.util.cleanup import cleanup as qonnx_cleanup

def main():
    device = torch.device("cpu")  # 导出时建议先用 CPU
    ckpt_path = ROOT / "best_qtcn_model.pth"
    out_path = ROOT / "ecg5000_qtcn_qonnx.onnx"

    # 这里的参数必须和训练时一致
    model = ECG5000QTCNClassifier(
        num_classes=5,
        seq_len=140,
        num_inputs=1,
        num_channels=(16, 16, 32),
        kernel_size=3,
        dropout=0.1,
        causal=False,
        use_skip_connections=False,
        input_shape='NCL',
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # ECG5000 输入形状: [N, C, L] = [1, 1, 140]
    dummy_input = torch.randn(1, 1, 140, device=device)

    export_qonnx(model, dummy_input, export_path=str(out_path), dynamo=False)
    print(f"QONNX exported to: {out_path}")


    qonnx_cleanup(str(out_path), out_file=str(out_path))
    print("QONNX cleanup done.")

if __name__ == "__main__":
    main()