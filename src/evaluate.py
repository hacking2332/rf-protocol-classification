"""
evaluate.py
-----------
Evaluation utilities:
  - Overall accuracy on test set
  - Per-SNR accuracy (the key benchmark for RF classification)
  - Confusion matrix (which modulations get confused)
  - Per-class precision / recall / F1
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from src.dataset import RadioMLDataset, MODULATIONS, IDX_TO_MOD


# ======================================================================
# Overall evaluation
# ======================================================================

@torch.no_grad()
def predict_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference over the full loader.

    Returns:
        y_true (N,), y_pred (N,)
    """
    model.eval()
    all_preds  = []
    all_labels = []

    for X, y in loader:
        X      = X.to(device, non_blocking=True)
        logits = model(X)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


def overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred)


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(
        y_true, y_pred,
        target_names=MODULATIONS,
        digits=3,
    )


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


# ======================================================================
# Per-SNR accuracy  (the main result to put in README)
# ======================================================================

@torch.no_grad()
def accuracy_per_snr(
    model: nn.Module,
    dataset: RadioMLDataset,
    device: torch.device,
    batch_size: int = 512,
) -> dict[int, float]:
    """
    Compute classification accuracy at each SNR level.

    Returns:
        dict  {snr (int): accuracy (float 0-1)}
    """
    model.eval()
    snr_levels = sorted(set(dataset.snrs.tolist()))
    results    = {}

    for snr in snr_levels:
        indices = dataset.get_snr_indices(snr)
        subset  = Subset(dataset, indices)
        loader  = DataLoader(subset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

        correct = 0
        total   = 0
        for X, y in loader:
            X      = X.to(device, non_blocking=True)
            logits = model(X)
            preds  = logits.argmax(dim=1).cpu()
            correct += (preds == y).sum().item()
            total   += y.size(0)

        results[snr] = correct / total
        print(f"  SNR {snr:+4d} dB  →  {results[snr]*100:.1f}%")

    avg = np.mean(list(results.values()))
    print(f"\n  Overall avg accuracy: {avg*100:.1f}%")
    return results


# ======================================================================
# Full evaluation report
# ======================================================================

def full_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    dataset: RadioMLDataset,
    device: torch.device,
) -> dict:
    """
    Run complete evaluation: overall accuracy, per-SNR, confusion matrix,
    classification report.

    Returns dict with all results.
    """
    print("\n" + "="*60)
    print("  EVALUATION REPORT")
    print("="*60)

    # -- Overall accuracy
    y_true, y_pred = predict_all(model, test_loader, device)
    acc = overall_accuracy(y_true, y_pred)
    print(f"\n  Test accuracy (all SNRs): {acc*100:.2f}%")

    # -- Classification report
    print("\n  Per-class metrics:")
    print(get_classification_report(y_true, y_pred))

    # -- Confusion matrix
    cm = get_confusion_matrix(y_true, y_pred)

    # -- Per-SNR accuracy
    print("\n  Accuracy per SNR level:")
    snr_acc = accuracy_per_snr(model, dataset, device)

    print("="*60)

    return {
        "accuracy":    acc,
        "y_true":      y_true,
        "y_pred":      y_pred,
        "conf_matrix": cm,
        "snr_acc":     snr_acc,
    }
