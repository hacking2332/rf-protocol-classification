"""
train.py
--------
Training loop with:
  - Adam optimizer + CosineAnnealingLR scheduler
  - Early stopping (patience-based)
  - Best model checkpointing
  - Per-epoch train/val loss and accuracy logging
  - tqdm progress bars
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ======================================================================
# Core train / eval functions
# ======================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler=None,                  # GradScaler for mixed precision (optional)
) -> tuple[float, float]:
    """
    Run one full training epoch.

    Returns:
        avg_loss (float), accuracy (float 0-1)
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for X, y in tqdm(loader, desc="  Train", leave=False, ncols=80):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(X)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y).sum().item()
        total      += X.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate on a DataLoader.

    Returns:
        avg_loss (float), accuracy (float 0-1)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for X, y in tqdm(loader, desc="  Val  ", leave=False, ncols=80):
        X, y   = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        loss   = criterion(logits, y)

        total_loss += loss.item() * X.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y).sum().item()
        total      += X.size(0)

    return total_loss / total, correct / total


# ======================================================================
# Early stopping helper
# ======================================================================

class EarlyStopping:
    """
    Stop training when val loss does not improve for `patience` epochs.
    Also saves the best model checkpoint.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 checkpoint_path: str = "models/best.pt"):
        self.patience         = patience
        self.min_delta        = min_delta
        self.checkpoint_path  = checkpoint_path
        self.best_loss        = float("inf")
        self.counter          = 0
        self.best_epoch       = 0

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Returns True if training should stop.
        Saves checkpoint if this is the best epoch.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss   = val_loss
            self.counter     = 0
            self.best_epoch  = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


# ======================================================================
# Main training function
# ======================================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int           = 50,
    lr: float             = 1e-3,
    weight_decay: float   = 1e-4,
    patience: int         = 10,
    checkpoint_path: str  = "models/best.pt",
    use_amp: bool         = True,
    device: torch.device  = None,
) -> dict:
    """
    Full training loop.

    Returns:
        history dict with keys: train_loss, val_loss, train_acc, val_acc
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # Mixed precision scaler — only useful on CUDA
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    early_stop = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    print(f"\n{'='*60}")
    print(f"  Training on : {device}")
    print(f"  Epochs      : {epochs}")
    print(f"  LR          : {lr}  |  Weight decay: {weight_decay}")
    print(f"  AMP         : {scaler is not None}")
    print(f"  Checkpoint  : {checkpoint_path}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train loss: {train_loss:.4f}  acc: {train_acc*100:.1f}% | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc*100:.1f}% | "
            f"LR: {lr_now:.2e} | {elapsed:.1f}s"
        )

        should_stop = early_stop(val_loss, model, epoch)

        if should_stop:
            print(f"\n[Early Stop] No improvement for {patience} epochs.")
            print(f"[Early Stop] Best epoch: {early_stop.best_epoch} | "
                  f"Best val loss: {early_stop.best_loss:.4f}")
            break

    print(f"\n[Done] Best checkpoint saved → {checkpoint_path}")
    return history
