import os
import warnings
import torch
import torch.nn as nn
import math

import brevitas.nn as qnn


from typing import Optional
from typing import Union
from typing import List

from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias
from brevitas.quant import Uint8ActPerTensorFloat

layer = qnn.QuantConvTranspose2d(
    in_channels=4,
    out_channels=8,
    kernel_size=(1, 4),
    stride=(1, 2),
    padding=(0, 1),
    output_padding=(0, 0),
    bias=True,
    weight_quant=Int8WeightPerTensorFloat,
    input_quant=Int8ActPerTensorFloat,
    bias_quant=Int32Bias,
    return_quant_tensor=False,
)
print(layer)