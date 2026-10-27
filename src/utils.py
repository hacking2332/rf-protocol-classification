"""
utils.py
--------
Helper functions:
  - Device setup
  - Reproducibility seed
  - Plot training history
  - Plot SNR accuracy curve
  - Plot confusion matrix
  - Plot I/Q signal and spectrogram
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import torch
from scipy.signal import stft

from src.dataset import MODULATIONS


# ======================================================================
# Setup helpers
# ======================================================================

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    return device


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[Seed] Set to {seed}")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ======================================================================
# Plotting: training history
# ======================================================================

def plot_training_history(history: dict, save_path: str = "results/training_history.png"):
    """Plot train/val loss and accuracy curves."""
    ensure_dir(os.path.dirname(save_path))

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   linewidth=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    train_acc_pct = [a * 100 for a in history["train_acc"]]
    val_acc_pct   = [a * 100 for a in history["val_acc"]]
    axes[1].plot(epochs, train_acc_pct, label="Train", linewidth=2)
    axes[1].plot(epochs, val_acc_pct,   label="Val",   linewidth=2, linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training history saved → {save_path}")


# ======================================================================
# Plotting: SNR accuracy curve
# ======================================================================

def plot_snr_accuracy(
    snr_results: dict,                     # {model_name: {snr: acc}}
    save_path: str = "results/snr_accuracy.png"
):
    """
    Plot accuracy vs SNR for one or multiple models on the same axes.

    Args:
        snr_results : dict of {model_name: {snr_int: acc_float}}
    """
    ensure_dir(os.path.dirname(save_path))

    fig, ax = plt.subplots(figsize=(10, 5))

    for model_name, snr_acc in snr_results.items():
        snrs = sorted(snr_acc.keys())
        accs = [snr_acc[s] * 100 for s in snrs]
        ax.plot(snrs, accs, marker="o", linewidth=2, label=model_name)

    ax.axhline(y=100 / len(MODULATIONS), color="gray", linestyle=":",
               label=f"Random ({100/len(MODULATIONS):.1f}%)")
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs SNR — RF Modulation Classification", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xticks(sorted(snr_results[list(snr_results.keys())[0]].keys()))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] SNR accuracy curve saved → {save_path}")


# ======================================================================
# Plotting: confusion matrix
# ======================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str = "Model",
    save_path: str  = "results/confusion_matrix.png",
    normalize: bool = True,
):
    """Plot a labeled confusion matrix heatmap."""
    ensure_dir(os.path.dirname(save_path))

    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt, vmax = ".2f", 1.0
    else:
        fmt, vmax = "d", None

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm,
        annot=True, fmt=fmt,
        xticklabels=MODULATIONS,
        yticklabels=MODULATIONS,
        cmap="Blues",
        vmin=0, vmax=vmax,
        linewidths=0.4,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Confusion matrix saved → {save_path}")


# ======================================================================
# Plotting: signal visualization
# ======================================================================

def plot_iq_samples(
    dataset,
    modulations: list = None,
    snr: int = 18,
    save_path: str = "results/iq_samples.png",
):
    """
    Plot raw I/Q waveforms for a selection of modulation types at a given SNR.
    """
    ensure_dir(os.path.dirname(save_path))

    if modulations is None:
        modulations = MODULATIONS[:6]

    from src.dataset import MOD_TO_IDX
    import pickle

    fig, axes = plt.subplots(len(modulations), 2, figsize=(12, 2.5 * len(modulations)))

    for i, mod in enumerate(modulations):
        # Find a sample for this mod+snr
        mask  = (dataset.snrs == snr)
        label_mask = (dataset.labels == MOD_TO_IDX[mod])
        indices = np.where(mask & label_mask)[0]

        if len(indices) == 0:
            continue

        sig = dataset.samples[indices[0]]      # (2, 128)
        t   = np.arange(128)

        axes[i, 0].plot(t, sig[0], color="#2563EB", linewidth=1.2)
        axes[i, 0].set_title(f"{mod} — I channel (SNR={snr} dB)", fontsize=9)
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(t, sig[1], color="#DC2626", linewidth=1.2)
        axes[i, 1].set_title(f"{mod} — Q channel (SNR={snr} dB)", fontsize=9)
        axes[i, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Raw I/Q Signals — {len(modulations)} Modulation Types at SNR = {snr} dB",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] I/Q samples saved → {save_path}")


def plot_spectrograms(
    dataset,
    modulations: list = None,
    snr: int = 18,
    save_path: str = "results/spectrograms.png",
):
    """Plot STFT spectrograms for selected modulation types."""
    ensure_dir(os.path.dirname(save_path))

    if modulations is None:
        modulations = MODULATIONS[:6]

    from src.dataset import MOD_TO_IDX

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()

    for i, mod in enumerate(modulations[:6]):
        mask       = (dataset.snrs == snr)
        label_mask = (dataset.labels == MOD_TO_IDX[mod])
        indices    = np.where(mask & label_mask)[0]

        if len(indices) == 0:
            continue

        sig         = dataset.samples[indices[0]]
        complex_sig = sig[0] + 1j * sig[1]
        f, t, Zxx  = stft(complex_sig, nperseg=32, noverlap=16)
        magnitude   = np.abs(Zxx)

        axes[i].imshow(
            np.log1p(magnitude),
            aspect="auto", origin="lower",
            cmap="viridis",
        )
        axes[i].set_title(f"{mod}  (SNR={snr} dB)", fontsize=10)
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Frequency")

    plt.suptitle("STFT Spectrograms — Log Magnitude", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Spectrograms saved → {save_path}")
