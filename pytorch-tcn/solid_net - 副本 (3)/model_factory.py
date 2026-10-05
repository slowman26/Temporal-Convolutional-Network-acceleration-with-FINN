from .ECG5000_QTCN_solid import ECG5000QTCNClassifier
from .quant_pad_solid import TemporalPad2dTrain, TemporalPad2dExport


def build_train_model(
    num_classes=5,
    seq_len=140,
    num_inputs=1,
    num_channels=(16, 16, 32),
    kernel_size=3,
    dropout=0.1,
    causal=True,
    use_skip_connections=False,
    input_shape="NCL",
):
    return ECG5000QTCNClassifier(
        num_classes=num_classes,
        seq_len=seq_len,
        num_inputs=num_inputs,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        causal=causal,
        use_skip_connections=use_skip_connections,
        input_shape=input_shape,
        padder_cls=TemporalPad2dTrain,
    )


def build_export_model(
    num_classes=5,
    seq_len=140,
    num_inputs=1,
    num_channels=(16, 16, 32),
    kernel_size=3,
    dropout=0.1,
    causal=True,
    use_skip_connections=False,
    input_shape="NCL",
):
    return ECG5000QTCNClassifier(
        num_classes=num_classes,
        seq_len=seq_len,
        num_inputs=num_inputs,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        causal=causal,
        use_skip_connections=use_skip_connections,
        input_shape=input_shape,
        padder_cls=TemporalPad2dExport,
    )