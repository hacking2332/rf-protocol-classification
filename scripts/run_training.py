"""
scripts/run_training.py
-----------------------
CLI entry point to train any model on the RadioML 2016.10a dataset.

Usage examples:
    python scripts/run_training.py --model cnn1d --epochs 50
    python scripts/run_training.py --model transformer --epochs 50 --lr 5e-4
    python scripts/run_training.py --model cnn2d --epochs 40 --batch_size 256
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset  import make_dataloaders
from src.models   import build_model
from src.train    import train
from src.evaluate import full_evaluation
from src.utils    import (
    get_device, set_seed,
    plot_training_history,
    plot_snr_accuracy,
    plot_confusion_matrix,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train RF modulation classifier")

    p.add_argument("--model",       type=str,   default="cnn1d",
                   choices=["cnn1d", "cnn2d", "transformer"],
                   help="Model architecture to train")
    p.add_argument("--data",        type=str,   default="RML2016.10a_dict.pkl",
                   help="Path to RadioML 2016.10a pickle file")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=512)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=10,
                   help="Early stopping patience (epochs)")
    p.add_argument("--snr_min",     type=int,   default=-20)
    p.add_argument("--snr_max",     type=int,   default=18)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--no_amp",      action="store_true",
                   help="Disable automatic mixed precision")
    return p.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    device = get_device()

    # -- Decide mode based on model
    mode = "spec" if args.model == "cnn2d" else "iq"

    # -- Load data
    train_loader, val_loader, test_loader, dataset = make_dataloaders(
        pkl_path   = args.data,
        mode       = mode,
        batch_size = args.batch_size,
        snr_min    = args.snr_min,
        snr_max    = args.snr_max,
    )

    # -- Build model
    model = build_model(args.model)

    # -- Train
    checkpoint = f"models/{args.model}_best.pt"
    history = train(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        epochs         = args.epochs,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        patience       = args.patience,
        checkpoint_path= checkpoint,
        use_amp        = not args.no_amp,
        device         = device,
    )

    # -- Load best checkpoint for evaluation
    import torch
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    # -- Evaluate
    results = full_evaluation(model, test_loader, dataset, device)

    # -- Save plots
    plot_training_history(history,
        save_path=f"results/{args.model}_training_history.png")

    plot_snr_accuracy(
        {args.model.upper(): results["snr_acc"]},
        save_path=f"results/{args.model}_snr_accuracy.png"
    )

    plot_confusion_matrix(
        results["conf_matrix"],
        model_name=args.model.upper(),
        save_path=f"results/{args.model}_confusion_matrix.png"
    )

    print(f"\n[Done] All results saved to results/")


if __name__ == "__main__":
    main()
