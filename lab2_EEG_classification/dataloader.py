from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"


def read_bci_data(return_subjects=False, fill_missing=True):
    """Training code fits imputation after splitting off validation data."""
    signals, labels, subjects = {}, {}, {}
    for split in ("train", "test"):
        xs, ys, ids = [], [], []
        for subject in ("S4b", "X11b"):
            with np.load(DATA_DIR / f"{subject}_{split}.npz", allow_pickle=False) as data:
                x, y = data["signal"], data["label"]
            if x.ndim != 3 or x.shape[1:] != (750, 2) or y.shape != (len(x),):
                raise ValueError(f"Unexpected shape: {subject}_{split}")
            if not np.isin(y, [1, 2]).all() or np.isinf(x).any():
                raise ValueError(f"Invalid label or infinite signal: {subject}_{split}")
            xs.append(x)
            ys.append(y.astype(np.int64) - 1)
            ids.extend([subject] * len(y))
        signals[split] = np.concatenate(xs).transpose(0, 2, 1)[:, None].astype(np.float32)
        labels[split] = np.concatenate(ys)
        subjects[split] = np.array(ids)
    if fill_missing:
        mean = float(np.nanmean(signals["train"]))
        if not np.isfinite(mean):
            raise ValueError("No finite training data for imputation")
        for x in signals.values():
            x[np.isnan(x)] = mean
    result = (signals["train"], labels["train"], signals["test"], labels["test"])
    return result + (subjects["train"], subjects["test"]) if return_subjects else result


def stratified_split(labels, subjects, seed=42):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for subject in np.unique(subjects):
        for label in np.unique(labels):
            indices = np.flatnonzero((subjects == subject) & (labels == label))
            if len(indices) < 5:
                raise ValueError("Each subject/class stratum needs at least five samples")
            indices = rng.permutation(indices)
            n_val = round(len(indices) * 0.2)
            val.extend(indices[:n_val])
            train.extend(indices[n_val:])
    return np.sort(train), np.sort(val)
