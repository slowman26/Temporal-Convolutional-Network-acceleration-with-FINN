#from pytorch_tcn.tcn import TCN
from pytorch_tcn.conv import TemporalConv1d
from pytorch_tcn.conv import TemporalConvTranspose1d

#from .tcn import TCN
from .conv import TemporalConv1d, TemporalConvTranspose1d

__all__ = [
    #"TCN",
    "TemporalConv1d",
    "TemporalConvTranspose1d",
]


__version__ = '1.2.2.dev1'