"""
dataset.py
----------
RadioML 2016.10a Dataset loader with:
  - Raw I/Q mode  (for CNN1D and Transformer)
  - Spectrogram mode (for CNN2D)
  - Train / Val / Test split with SNR filtering
"""

import pickle
import numpy as np
from scipy.signal import stft
from torch.utils.data import Dataset, DataLoader, random_split
import torch


# 11 modulation classes in RadioML 2016.10a
MODULATIONS = [
    '8PSK', 'AM-DSB', 'AM-SSB', 'BPSK',
    'CPFSK', 'GFSK', 'PAM4', 'QAM16',
    'QAM64', 'QPSK', 'WBFM'
]
NUM_CLASSES = len(MODULATIONS)
MOD_TO_IDX  = {m: i for i, m in enumerate(MODULATIONS)}
IDX_TO_MOD  = {i: m for m, i in MOD_TO_IDX.items()}


def load_radioml(pkl_path: str):
    """Load the raw RadioML 2016.10a pickle file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


class RadioMLDataset(Dataset):
    """
    PyTorch Dataset for RadioML 2016.10a.

    Args:
        pkl_path  : path to RML2016.10a_dict.pkl
        mode      : 'iq'   -> returns (2, 128) raw I/Q tensor
                    'spec' -> returns (1, F, T) STFT magnitude spectrogram
        snr_min   : minimum SNR to include (default -20 dB)
        snr_max   : maximum SNR to include (default +18 dB)
        normalize : zero-mean unit-variance normalization per sample
    """

    def __init__(
        self,
        pkl_path: str,
        mode: str = "iq",
        snr_min: int = -20,
        snr_max: int = 18,
        normalize: bool = True,
    ):
        assert mode in ("iq", "spec"), "mode must be 'iq' or 'spec'"
        self.mode      = mode
        self.normalize = normalize

        raw = load_radioml(pkl_path)

        samples, labels, snrs = [], [], []
        for (mod, snr), signals in raw.items():
            if snr_min <= snr <= snr_max and mod in MOD_TO_IDX:
                for sig in signals:
                    samples.append(sig.astype(np.float32))   # (2, 128)
                    labels.append(MOD_TO_IDX[mod])
                    snrs.append(snr)

        self.samples = np.array(samples)   # (N, 2, 128)
        self.labels  = np.array(labels,  dtype=np.int64)
        self.snrs    = np.array(snrs,    dtype=np.int32)

        print(f"[Dataset] Loaded {len(self.samples):,} samples "
              f"| {NUM_CLASSES} classes | SNR {snr_min} to {snr_max} dB")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sig   = self.samples[idx]          # (2, 128)
        label = self.labels[idx]

        if self.mode == "iq":
            x = self._process_iq(sig)
        else:
            x = self._process_spec(sig)

        return x, torch.tensor(label, dtype=torch.long)

    # ------------------------------------------------------------------
    def _process_iq(self, sig: np.ndarray) -> torch.Tensor:
        """Return normalized (2, 128) tensor."""
        if self.normalize:
            sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        return torch.tensor(sig, dtype=torch.float32)

    def _process_spec(self, sig: np.ndarray) -> torch.Tensor:
        """
        Compute STFT spectrogram from I/Q and return (1, F, T) tensor.
        Complex signal = I + jQ  →  |STFT|  log-scaled.
        """
        complex_sig = sig[0] + 1j * sig[1]
        _, _, Zxx = stft(complex_sig, nperseg=32, noverlap=16)
        magnitude = np.abs(Zxx).astype(np.float32)          # (F, T)
        log_mag   = np.log1p(magnitude)                       # log scale

        if self.normalize:
            log_mag = (log_mag - log_mag.mean()) / (log_mag.std() + 1e-8)

        return torch.tensor(log_mag, dtype=torch.float32).unsqueeze(0)  # (1,F,T)

    # ------------------------------------------------------------------
    def get_snr_indices(self, target_snr: int) -> np.ndarray:
        """Return indices of all samples at a given SNR level."""
        return np.where(self.snrs == target_snr)[0]

    def input_shape(self) -> tuple:
        x, _ = self[0]
        return tuple(x.shape)


# ------------------------------------------------------------------
def make_dataloaders(
    pkl_path: str,
    mode: str = "iq",
    batch_size: int = 512,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    snr_min: int = -20,
    snr_max: int = 18,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, RadioMLDataset]:
    """
    Build train / val / test DataLoaders from a single pkl file.

    Returns:
        train_loader, val_loader, test_loader, full_dataset
    """
    dataset = RadioMLDataset(
        pkl_path, mode=mode, snr_min=snr_min, snr_max=snr_max
    )

    n        = len(dataset)
    n_test   = int(n * test_ratio)
    n_val    = int(n * val_ratio)
    n_train  = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"[DataLoader] Train: {n_train:,}  Val: {n_val:,}  Test: {n_test:,}")
    return train_loader, val_loader, test_loader, dataset
