#from pytorch_tcn.tcn import TCN
from pytorch_tcn.conv import TemporalConv1d
from pytorch_tcn.conv import TemporalConvTranspose1d

#from quant_tcn.test_model import ToyQuantTCN

from quant_tcn.quant_tcn_test import QTCN
from quant_tcn.ECG5000_QTCN import ECG5000QTCNClassifier

__version__ = '1.2.2.dev1'