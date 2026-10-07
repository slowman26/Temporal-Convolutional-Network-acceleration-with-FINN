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

## Deployed FPGA Results

The generated accelerators were deployed and evaluated on a **PYNQ-Z2 (XC7Z020)** FPGA.

### Quantization and Inference Performance

| Configuration | Platform | Accuracy | Batch | Latency / Sample (ms) | Throughput (FPS) |
|---|---|---:|---:|---:|---:|
| W8A8 | FPGA | 92.53% | 1 | 5.043 | 198.29 |
| W4A4 | FPGA | 91.16% | 1 | 4.979 | 200.86 |
| W2A4 | FPGA | 89.98% | 1 | 4.959 | 201.63 |
| W4A4 Folding 1 | FPGA | 91.16% | 1 | 4.151 | 240.88 |
| W4A4 Folding 2 | FPGA | 91.16% | 1 | 3.994 | 250.39 |
| W4A4 Folding 3 | FPGA | 91.16% | 1 | **3.881** | **257.69** |
| W4A4 Folding 4 | FPGA | 91.16% | 1 | 3.928 | 254.56 |
| W4A4 Folding 4 | FPGA | 91.16% | 8 | 1.859 | 537.92 |
| W4A4 Folding 4 | FPGA | 91.16% | 16 | 1.711 | 584.45 |
| W4A4 Folding 4 | FPGA | 91.16% | 348 | **1.592** | **628.11** |

For batch size 1, **Folding 3 achieved the lowest measured latency of 3.881 ms per sample**.  
With batching, Folding 4 reached an effective throughput of **628.11 samples/s** at batch size 348.

### FPGA Resource Utilization

Resource utilization for the W4A4 accelerator under different folding configurations:

| Configuration | LUT | LUT Util. | FF | FF Util. | BRAM 36K + 18K | BRAM18K Eq. | DSP | DSP Util. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 16,804 | 31.6% | 25,533 | 24.0% | 3 + 9 | 15 | 10 | 4.5% |
| Folding 1 | 19,543 | 36.7% | 26,516 | 24.9% | 3 + 9 | 15 | 26 | 11.8% |
| Folding 2 | 21,525 | 40.5% | 27,649 | 26.0% | 3 + 9 | 15 | 46 | 20.9% |
| Folding 3 | 24,448 | 46.0% | 27,893 | 26.2% | 7 + 4 | 18 | 86 | 39.1% |
| Folding 4 | 25,468 | 47.9% | 29,175 | 27.4% | 12 + 5 | 29 | 167 | 75.9% |

PYNQ-Z2 / XC7Z020 resources:

- LUTs: 53,200
- FFs: 106,400
- BRAM: 140 × 36-Kbit
- DSPs: 220

Increasing PE/SIMD parallelism reduced the accelerator bottleneck but increased resource usage, especially DSP utilization. Folding 4 used 167 DSP blocks (75.9%), while Folding 3 required only 86 DSPs (39.1%).

### Post-Route Timing

All evaluated W4A4 implementations met the 100 MHz timing target.

| Configuration | WNS (ns) | TNS (ns) | WHS (ns) | THS (ns) | Timing Met |
|---|---:|---:|---:|---:|---|
| Baseline | 0.208 | 0.000 | 0.008 | 0.000 | Yes |
| Folding 1 | 0.244 | 0.000 | 0.012 | 0.000 | Yes |
| Folding 2 | 0.316 | 0.000 | 0.023 | 0.000 | Yes |
| Folding 3 | 0.201 | 0.000 | 0.026 | 0.000 | Yes |
| Folding 4 | 0.610 | 0.000 | 0.022 | 0.000 | Yes |

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


