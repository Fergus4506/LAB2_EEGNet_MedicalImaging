"""Train one experiment or the eight predeclared Lab2 experiments."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path

# Required by deterministic CUDA matrix multiplication before CUDA initializes.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataloader import DATA_DIR, read_bci_data, stratified_split
from models.EEGNet import DeepConvNet, EEGNet

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = [
    ("eegnet_baseline", "eegnet", "elu", 1.0, 0.25),
    ("eegnet_relu", "eegnet", "relu", 1.0, 0.25),
    ("eegnet_leakyrelu", "eegnet", "leakyrelu", 1.0, 0.25),
    ("eegnet_alpha05", "eegnet", "elu", 0.5, 0.25),
    ("eegnet_alpha2", "eegnet", "elu", 2.0, 0.25),
    ("eegnet_dropout0", "eegnet", "elu", 1.0, 0.0),
    ("eegnet_dropout05", "eegnet", "elu", 1.0, 0.5),
    ("deepconvnet_baseline", "deepconvnet", "elu", 1.0, 0.5),
]


class BCIDataset(Dataset):
    def __init__(self, data, label):
        if data.shape != (len(label), 1, 2, 750) or not np.isfinite(data).all():
            raise ValueError("Expected finite data with shape (N, 1, 2, 750)")
        if not np.isin(label, [0, 1]).all():
            raise ValueError("Labels must be 0 or 1")
        self.data = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
        self.label = torch.from_numpy(np.asarray(label, dtype=np.int64))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def make_model(config):
    cls = EEGNet if config["model"] == "eegnet" else DeepConvNet
    return cls(config["activation"], config["alpha"], config["dropout"])


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; experiment stopped")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            count += len(labels)
    if count == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return total_loss / count, 100.0 * correct / count


def better_checkpoint(acc, loss, best_acc, best_loss):
    return acc > best_acc or (acc == best_acc and loss < best_loss)


def plot_series(values, ylabel, output):
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for label, series in values.items():
        ax.plot(range(1, len(series) + 1), series, label=label, linewidth=1.2)
    if ylabel == "Cross-entropy loss":
        ax.set_yscale("log")
        ylabel += " (log scale)"
    ax.set(xlabel="Epoch", ylabel=ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_train_acc(train_acc_list, epochs, output="train_accuracy.png"):
    plot_series({"Train": train_acc_list[:epochs]}, "Accuracy (%)", output)


def plot_train_loss(train_loss_list, epochs, output="train_loss.png"):
    plot_series({"Train": train_loss_list[:epochs]}, "Cross-entropy loss", output)


def plot_test_acc(test_acc_list, epochs, output="test_accuracy.png"):
    plot_series({"Test": test_acc_list[:epochs]}, "Accuracy (%)", output)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def run_experiment(config, output):
    output.mkdir(parents=True, exist_ok=False)
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, tx, ty, subjects, _ = read_bci_data(return_subjects=True, fill_missing=False)
    train_idx, val_idx = stratified_split(y, subjects, config["seed"])
    mean = float(np.nanmean(x[train_idx]))
    if not math.isfinite(mean):
        raise ValueError("Training imputation mean is not finite")
    for data in (x, tx):
        data[np.isnan(data)] = mean
    np.savez(output / "split.npz", train_indices=train_idx, validation_indices=val_idx)
    config = dict(config, imputation_mean=mean,
                  train_samples=len(train_idx), validation_samples=len(val_idx), test_samples=len(ty),
                  dataset_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(DATA_DIR.glob("*.npz"))},
                  environment={"python": platform.python_version(), "torch": str(torch.__version__),
                               "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                               "cuda": torch.version.cuda, "device": str(device),
                               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None})
    write_json(output / "config.json", config)
    datasets = [BCIDataset(x[train_idx], y[train_idx]), BCIDataset(x[val_idx], y[val_idx]), BCIDataset(tx, ty)]
    loaders = [DataLoader(ds, batch_size=config["batch_size"], shuffle=(i == 0),
                          generator=torch.Generator().manual_seed(config["seed"]))
               for i, ds in enumerate(datasets)]
    model = make_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0.001)
    best_acc, best_loss, best_epoch = -1.0, float("inf"), 0
    history = []
    started = time.perf_counter()
    log = (output / "train.log").open("w", encoding="utf-8", buffering=1)
    try:
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"]
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            for epoch in range(1, config["num_epochs"] + 1):
                tr_loss, tr_acc = run_epoch(model, loaders[0], criterion, device, optimizer)
                val_loss, val_acc = run_epoch(model, loaders[1], criterion, device)
                test_loss, test_acc = run_epoch(model, loaders[2], criterion, device)
                row = dict(zip(fields, [epoch, tr_loss, tr_acc, val_loss, val_acc, test_loss, test_acc]))
                history.append(row)
                writer.writerow(row)
                handle.flush()
                if better_checkpoint(val_acc, val_loss, best_acc, best_loss):
                    best_acc, best_loss, best_epoch = val_acc, val_loss, epoch
                    # Clone tensors so later optimizer steps cannot change selected weights.
                    weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    torch.save({"state_dict": weights, "config": config, "epoch": epoch,
                                "val_acc": val_acc, "val_loss": val_loss}, output / "best.pt")
                line = (f'{config["name"]} epoch={epoch:03d} train_loss={tr_loss:.6f} '
                        f'train_acc={tr_acc:.4f}% val_loss={val_loss:.6f} val_acc={val_acc:.4f}% '
                        f'test_loss={test_loss:.6f} test_acc={test_acc:.4f}%')
                log.write(line + "\n")
                if epoch == 1 or epoch % 10 == 0 or epoch == config["num_epochs"]:
                    print(line, flush=True)
    finally:
        log.close()
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    selected_loss, selected_acc = run_epoch(model, loaders[2], criterion, device)
    recorded = history[best_epoch - 1]
    if abs(selected_acc - recorded["test_acc"]) > 1e-8 or abs(selected_loss - recorded["test_loss"]) > 1e-5:
        raise AssertionError("Reloaded checkpoint does not reproduce recorded evaluation")
    for metric, ylabel in (("acc", "Accuracy (%)"), ("loss", "Cross-entropy loss")):
        plot_series({split.title(): [r[f"{split}_{metric}"] for r in history]
                     for split in ("train", "val", "test")}, ylabel, output / f"{metric}.png")
    peak = max(history, key=lambda r: r["test_acc"])
    result = {"name": config["name"], "model": config["model"], "activation": config["activation"],
              "alpha": config["alpha"], "dropout": config["dropout"], "best_epoch": best_epoch,
              "val_acc": best_acc, "val_loss": best_loss, "selected_test_acc": selected_acc,
              "selected_test_loss": selected_loss, "peak_test_acc": peak["test_acc"],
              "peak_test_epoch": peak["epoch"], "parameters": sum(p.numel() for p in model.parameters()),
              "seconds": time.perf_counter() - started}
    write_json(output / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def evaluate_checkpoint(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    set_seed(config["seed"])
    for filename, expected in config["dataset_sha256"].items():
        if hashlib.sha256((DATA_DIR / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dataset changed: {filename}")
    _, _, tx, ty = read_bci_data(fill_missing=False)
    tx[np.isnan(tx)] = config["imputation_mean"]
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    loss, acc = run_epoch(model, DataLoader(BCIDataset(tx, ty), batch_size=config["batch_size"]),
                          nn.CrossEntropyLoss(), device)
    result = {"checkpoint": str(path.resolve()), "epoch": checkpoint["epoch"],
              "test_loss": loss, "test_acc": acc}
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-num_epochs", "--num-epochs", type=int, default=150)
    parser.add_argument("-batch_size", "--batch-size", type=int, default=64)
    parser.add_argument("-lr", "--lr", type=float, default=0.01)
    parser.add_argument("--model", choices=["eegnet", "deepconvnet"], default="eegnet")
    parser.add_argument("--activation", choices=["elu", "relu", "leakyrelu"], default="elu")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--evaluate", type=Path, metavar="CHECKPOINT")
    args = parser.parse_args()
    if args.num_epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, batch size, and finite learning rate must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    if args.evaluate:
        evaluate_checkpoint(args.evaluate)
        return
    base = {k: getattr(args, k) for k in ("num_epochs", "batch_size", "lr", "seed")}
    if args.suite:
        experiments = EXPERIMENTS
    else:
        dropout = args.dropout if args.dropout is not None else (0.5 if args.model == "deepconvnet" else 0.25)
        experiments = [("single", args.model, args.activation, args.alpha, dropout)]
    if any((args.output / e[0]).exists() for e in experiments):
        parser.error("Experiment directory exists; choose a new --output to preserve earlier results")
    results = []
    for name, model, activation, alpha, dropout in experiments:
        config = dict(base, name=name, model=model, activation=activation, alpha=alpha, dropout=dropout)
        results.append(run_experiment(config, args.output / name))
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
