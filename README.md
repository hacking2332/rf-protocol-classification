# RF Protocol Classification — Deep Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-RadioML%202016.10a-orange)

An end-to-end deep learning pipeline for **automatic modulation classification (AMC)** of radio frequency signals.  
Classifies 11 modulation types from raw 2-channel **In-phase / Quadrature (I/Q)** time-series data across signal-to-noise ratios ranging from **−20 dB to +18 dB**.

---

## Results

| Model | Parameters | Avg Accuracy | High SNR (≥0 dB) | Inference Latency |
|---|---|---|---|---|
| **1D-CNN** | 214K | 57.3% | 83.6% | 0.8 ms |
| **2D-CNN (Spectrogram)** | 318K | 55.1% | 80.2% | 1.1 ms |
| **Transformer** | 187K | 59.8% | 86.1% | 1.4 ms |

> Accuracy drops at low SNR (−20 dB to −6 dB) — consistent with published RadioML benchmarks.  
> All models evaluated on 30% held-out test split, 11-class classification.

---

## Modulation Types

The pipeline classifies 11 modulation types from the **RadioML 2016.10a** dataset:

`8PSK` · `AM-DSB` · `AM-SSB` · `BPSK` · `CPFSK` · `GFSK` · `PAM4` · `QAM16` · `QAM64` · `QPSK` · `WBFM`

---

## Architecture Overview

```
Raw I/Q Signal (2 × 128)
        │
        ├──► 1D-CNN ──────────────────► Conv1d layers → FC → Softmax
        │
        ├──► 2D-CNN (Spectrogram) ────► STFT → Conv2d layers → FC → Softmax
        │
        └──► Transformer ─────────────► Linear projection → Encoder layers → GlobalAvgPool → Softmax
```

### 1D-CNN Architecture
```
Input (B, 2, 128)
  → Conv1d(2→64, k=3) → ReLU → Conv1d(64→64, k=3) → ReLU → MaxPool1d(2)
  → Conv1d(64→128, k=3) → ReLU → Conv1d(128→128, k=3) → ReLU → MaxPool1d(2)
  → Conv1d(128→256, k=3) → ReLU → AdaptiveAvgPool1d(1)
  → Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→11)
```

### Transformer Architecture
```
Input (B, 2, 128)
  → Permute → (B, 128, 2)
  → Linear(2→64)  [input projection]
  → TransformerEncoder [2 layers, 4 heads, d_ff=128]
  → GlobalAveragePool
  → Linear(64→11)
```

---

## Project Structure

```
rf-protocol-classification/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── dataset.py          # RadioML Dataset class + preprocessing
│   ├── models.py           # CNN1D, CNN2D, RFTransformer architectures
│   ├── train.py            # Training loop + LR scheduler + early stopping
│   ├── evaluate.py         # Accuracy per SNR, confusion matrix, metrics
│   └── utils.py            # Plotting helpers, seed, device setup
│
├── scripts/
│   ├── run_training.py     # Entry point: train any model from CLI
│   ├── run_benchmark.py    # Inference latency benchmarking
│   └── export_onnx.py      # Export trained model to ONNX
│
├── notebooks/
│   └── exploration.ipynb   # Data exploration + results visualization
│
├── models/                 # Saved .pt checkpoints (git-ignored)
└── results/                # SNR curves, confusion matrices (generated)
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/hacking2332/rf-protocol-classification.git
cd rf-protocol-classification
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
# Free download from DeepSig
wget http://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2
tar -xjf RML2016.10a.tar.bz2
```

### 3. Train a model

```bash
# Train 1D-CNN (fastest)
python scripts/run_training.py --model cnn1d --epochs 50 --batch_size 512

# Train Transformer
python scripts/run_training.py --model transformer --epochs 50 --batch_size 256

# Train 2D-CNN on spectrograms
python scripts/run_training.py --model cnn2d --epochs 50 --batch_size 256
```

### 4. Benchmark inference latency

```bash
python scripts/run_benchmark.py --model cnn1d --checkpoint models/cnn1d_best.pt
```

### 5. Export to ONNX

```bash
python scripts/export_onnx.py --model transformer --checkpoint models/transformer_best.pt
```

---

## Run on Google Colab (No GPU Required Locally)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Open `notebooks/exploration.ipynb` in Google Colab for a full walkthrough with plots and results.

---

## Background: What is I/Q Data?

Radio signals are represented as two components:
- **I (In-phase)** — the real part of the complex baseband signal
- **Q (Quadrature)** — the imaginary part, 90° shifted

Each sample in the dataset is a `(2, 128)` array — 128 time steps of both channels.  
Different modulation schemes (BPSK, QAM16, etc.) create distinct patterns in I/Q space,  
which the CNN and Transformer learn to distinguish — even under heavy noise.

### Why is this hard?
At low SNR (e.g. −20 dB), noise overwhelms the signal. Even expert systems struggle below −6 dB.  
Deep learning models can learn noise-robust representations that beat hand-crafted features.

---

## Key Technical Decisions

| Decision | Choice | Reason |
|---|---|---|
| Input format for CNN1D | Raw I/Q (2, 128) | No preprocessing needed, end-to-end learning |
| Input format for CNN2D | STFT spectrogram | Captures time-frequency structure |
| Transformer positional encoding | Learned | Better than fixed for short 128-step sequences |
| Loss function | CrossEntropyLoss | Standard for multi-class classification |
| Optimizer | Adam (lr=1e-3) | Faster convergence than SGD for this task |
| Evaluation metric | Accuracy vs SNR curve | Industry-standard for AMC benchmarking |

---

## References

1. O'Shea, T., Corgan, J., Clancy, T. (2016). *Convolutional Radio Modulation Recognition Networks*. [arXiv:1602.04105](https://arxiv.org/abs/1602.04105)
2. DeepSig Inc. *RadioML Datasets*. [opendata.deepsig.io](http://opendata.deepsig.io)
3. brysef/rfml — RFML Library in PyTorch. [github.com/brysef/rfml](https://github.com/brysef/rfml)

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Built as part of ongoing research into edge-deployable RF signal intelligence systems.*
