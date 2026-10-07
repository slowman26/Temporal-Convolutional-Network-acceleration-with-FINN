# Temporal Convolutional Network Acceleration with FINN

> FPGA acceleration of quantized causal Temporal Convolutional Networks using
> FINN on the Xilinx PYNQ-Z2 platform.

**Master's Thesis — Linköping University**

# Temporal Convolutional Network Acceleration with FINN

This repository contains the implementation of my Master's thesis project on accelerating quantized Temporal Convolutional Networks (TCNs) on FPGA using the [FINN](https://github.com/Xilinx/finn) framework.

The work focuses on adapting a causal 1D TCN for FINN-compatible dataflow acceleration and evaluating different quantization configurations on a Xilinx PYNQ-Z2 FPGA.

## Overview

Temporal Convolutional Networks are widely used for sequence modelling, but mapping causal 1D convolutions to FPGA-oriented dataflow frameworks such as FINN requires several architectural and graph-level adaptations.

This project develops a FINN-compatible implementation of a quantized causal TCN for ECG classification.

The main contributions include:

- Reformulating 1D temporal convolutions as FINN-compatible pseudo-2D convolutions.
- Replacing the flatten + linear classifier with a convolutional classifier.
- Supporting causal padding in the FINN dataflow pipeline.
- Preserving integer datatypes during graph transformations.
- Absorbing scalar operations into threshold layers where possible.
- Generating complete FINN dataflow accelerators for PYNQ-Z2.
- Evaluating multiple weight/activation quantization configurations.
- Exploring folding configurations and FPGA resource/performance trade-offs.

## Target Application

The experiments use the **ECG5000** dataset for five-class ECG time-series classification.

- Number of classes: 5
- Sequence length: 140
- Input channels: 1

The TCN consists of three temporal convolutional stages followed by a convolution-based classifier.

## Quantization

The following quantization configurations were investigated:

| Configuration | Weights | Activations |
|---|---:|---:|
| W8A8 | 8-bit | 8-bit |
| W4A4 | 4-bit | 4-bit |
| W2A4 | 2-bit | 4-bit |

W4A4 provides a good trade-off between classification accuracy and FPGA resource usage.

## Hardware Platform

The generated accelerator targets:

- **Board:** PYNQ-Z2
- **FPGA:** Xilinx Zynq-7020
- **Framework:** FINN
- **Deployment:** FINN dataflow accelerator with Zynq I/O DMA

## Repository Structure

```text
.
├── datasets/
│   └── Dataset files used in the experiments
│
├── new_version/
│   └── Updated model and experiment implementations
│
├── onnx2bit/
│   └── ONNX / quantized model related experiments
│
├── pytorch-tcn/
│   ├── quant_tcn/
│   ├── pytorch_tcn/
│   ├── datasets/
│   ├── onnx_exports/
│   ├── build_dataflow.py
│   ├── absorb.py
│   └── ...
│
├── tcn2pynq/
│   └── Files related to PYNQ deployment and validation
│
└── LICENSE
