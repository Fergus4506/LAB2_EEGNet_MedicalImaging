"""Small checks for data splitting, metrics, gradients and checkpoint reload."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataloader import read_bci_data, stratified_split
from main import (ROOT, BCIDataset, better_checkpoint, evaluate_checkpoint,
                  run_epoch, run_experiment, set_seed, augment_batch, normalize_data, evaluate_ensemble,
                  aggregate_results)
from models.EEGNet import EEGNet, DeepConvNet


def main():
    set_seed(42)
    original = Path.cwd()
    try:
        os.chdir(ROOT / "models")
        x, y, tx, ty, subjects, _ = read_bci_data(return_subjects=True, fill_missing=False, train_only=True)
    finally:
        os.chdir(original)
    train, val = stratified_split(y, subjects)
    assert (len(train), len(val), len(ty)) == (864, 216, 0)
    assert not np.intersect1d(train, val).size
    assert np.array_equal(np.sort(np.r_[train, val]), np.arange(1080))
    assert all(np.array_equal(a, b) for a, b in zip((train, val), stratified_split(y, subjects)))
    for indices, n in ((train, 216), (val, 54)):
        for subject in np.unique(subjects):
            for label in (0, 1):
                assert np.sum((subjects[indices] == subject) & (y[indices] == label)) == n
    mean = float(np.nanmean(x[train]))
    for data in (x, tx):
        data[np.isnan(data)] = mean
        assert np.isfinite(data).all()
    ds = BCIDataset(x, y)
    assert ds[0][0].dtype == torch.float32 and ds[0][1].dtype == torch.int64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for mode, count in (("depthwise", 4018), ("full", 17874)):
        model = EEGNet(separable=mode).to(device)
        assert sum(p.numel() for p in model.parameters()) == count
        loss = nn.CrossEntropyLoss()(model(ds.data[:2].to(device)), ds.label[:2].to(device))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    batch = ds.data[:5].to(device)
    a = augment_batch(batch, torch.Generator(device=device).manual_seed(42))
    b = augment_batch(batch, torch.Generator(device=device).manual_seed(42))
    assert torch.equal(a, b) and torch.isfinite(a).all() and not torch.equal(a, batch)
    for cls in (EEGNet, DeepConvNet):
        for activation in ("elu", "relu", "leakyrelu"):
            model = cls(activation).to(device)
            logits = model(ds.data[:2].to(device))
            assert logits.shape == (2, 2)
            loss = nn.CrossEntropyLoss()(logits, ds.label[:2].to(device))
            loss.backward()
            assert torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for kwargs in ({"dropout": -0.1}, {"dropout": 1}, {"alpha": float("nan")}, {"activation": "bad"}):
        try:
            EEGNet(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid options: {kwargs}")
    logits = torch.tensor([[6., -2.], [1., 2.], [-1., 3.], [5., -2.], [0., 0.]])
    labels = torch.tensor([0, 0, 1, 1, 1])
    expected = nn.CrossEntropyLoss()(logits, labels).item()
    actual, acc = run_epoch(nn.Identity(), DataLoader(TensorDataset(logits, labels), batch_size=2),
                            nn.CrossEntropyLoss(), torch.device("cpu"))
    assert abs(actual - expected) < 1e-6, (actual, expected)
    assert acc == 40.0
    assert better_checkpoint(80, 0.4, 79, 0.1)
    assert better_checkpoint(80, 0.4, 80, 0.5)
    assert not better_checkpoint(80, 0.5, 80, 0.5)
    assert not better_checkpoint(79, 0.1, 80, 0.5)
    aggregate = aggregate_results([dict(seed=42, val_acc=80), dict(seed=43, val_acc=82)])
    assert aggregate["val_acc_mean"] == 81 and abs(aggregate["val_acc_std"] - np.sqrt(2)) < 1e-10
    with patch("main.checkpoint_probabilities", side_effect=[
        (np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([0, 1]), 0, 1),
        (np.array([[0.4, 0.6], [0.6, 0.4]]), np.array([0, 1]), 0, 1),
    ]):
        combined = evaluate_ensemble([Path("a"), Path("b")])
    assert combined["test_acc"] == 100 and abs(combined["test_loss"] + np.log([0.65, 0.6]).mean()) < 1e-10
    cache = ROOT.parent / ".cache"
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as temp:
        output = Path(temp) / "smoke"
        result = run_experiment(dict(name="selfcheck", model="eegnet", activation="elu", alpha=1.0,
                                     dropout=0.25, seed=42, num_epochs=2, batch_size=64, lr=0.01,
                                     separable="full", normalize=True, augment=True, optimizer="adamw",
                                     scheduler="cosine", validation_only=True), output)
        assert "selected_test_acc" not in result
        print("Evaluation fixture uses held-out validation data, never actual test samples")
        with patch("main.read_bci_data", return_value=(x, y, x[val].copy(), y[val])):
            evaluation = evaluate_checkpoint(output / "best.pt")
            ensemble = evaluate_ensemble([output / "best.pt", output / "best.pt"])
        assert abs(evaluation["test_acc"] - result["val_acc"]) < 1e-8
        assert abs(evaluation["test_loss"] - result["val_loss"]) < 1e-5
        assert ensemble["test_acc"] == evaluation["test_acc"]
        checkpoint = torch.load(output / "best.pt", weights_only=True)
        normalized = normalize_data(x[train], checkpoint["config"])
        assert np.allclose(checkpoint["config"]["normalization_mean"], x[train].mean(axis=(0, 1, 3), dtype=np.float64))
        assert np.allclose(checkpoint["config"]["normalization_std"], x[train].std(axis=(0, 1, 3), dtype=np.float64))
        assert np.allclose(normalized.mean(axis=(0, 1, 3), dtype=np.float64), 0, atol=1e-5)
        assert np.allclose(normalized.std(axis=(0, 1, 3), dtype=np.float64), 1, atol=1e-5)
        repeat = run_experiment(checkpoint["config"], Path(temp) / "repeat")
        assert repeat["val_acc"] == result["val_acc"] and repeat["val_loss"] == result["val_loss"]
        repeated = torch.load(Path(temp) / "repeat" / "best.pt", weights_only=True)
        assert all(torch.equal(v, repeated["state_dict"][k]) for k, v in checkpoint["state_dict"].items())
        full = run_experiment(dict(checkpoint["config"], full_train=True), Path(temp) / "full")
        assert full["selection"] == "fixed_epochs_full_train" and full["best_epoch"] == 2
        assert full["val_acc"] is None and full["reload_verified"]
        assert all((output / name).is_file() for name in
                   ("config.json", "history.csv", "best.pt", "split.npz", "summary.json", "acc.png", "loss.png"))
        before = (output / "history.csv").read_bytes()
        try:
            run_experiment({}, output)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing experiment should not be overwritten")
        assert (output / "history.csv").read_bytes() == before
    print(f"PASS: data, split, weighted loss, gradients, selection, reload, path and overwrite checks ({device})")


if __name__ == "__main__":
    main()
