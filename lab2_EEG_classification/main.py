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
    options = {"separable": config.get("separable", "depthwise")} if cls is EEGNet else {}
    return cls(config["activation"], config["alpha"], config["dropout"], **options)


def augment_batch(inputs, generator):
    shifts = torch.randint(-50, 51, (len(inputs), 1, 1, 1), device=inputs.device, generator=generator)
    indices = (torch.arange(inputs.shape[-1], device=inputs.device) - shifts) % inputs.shape[-1]
    shifted = inputs.gather(-1, indices.expand_as(inputs))
    scale = 0.9 + 0.2 * torch.rand((len(inputs), 1, 1, 1), device=inputs.device, generator=generator)
    noise = torch.randn(inputs.shape, device=inputs.device, generator=generator)
    return shifted * scale + noise * (0.05 * inputs.std(dim=-1, keepdim=True, unbiased=False))


def normalize_data(data, config):
    if config.get("normalize", False):
        mean = np.asarray(config["normalization_mean"], dtype=np.float32).reshape(1, 1, 2, 1)
        std = np.asarray(config["normalization_std"], dtype=np.float32).reshape(1, 1, 2, 1)
        data = (data - mean) / std
    return data


def run_epoch(model, loader, criterion, device, optimizer=None, augmentation_generator=None):
    model.train(optimizer is not None)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if optimizer is not None and augmentation_generator is not None:
                inputs = augment_batch(inputs, augmentation_generator)
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


DEFAULTS = dict(separable="depthwise", weight_decay=0.001, optimizer="adam", scheduler="none",
                normalize=False, augment=False, full_train=False, validation_only=False, split_seed=42)


def run_experiment(config, output):
    output.mkdir(parents=True, exist_ok=False)
    config = DEFAULTS | config
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, _, _, subjects, _ = read_bci_data(return_subjects=True, fill_missing=False, train_only=True)
    train_idx, val_idx = stratified_split(y, subjects, config["split_seed"])
    if config["full_train"]:
        train_idx, val_idx = np.arange(len(y)), np.empty(0, dtype=np.int64)
    mean = float(np.nanmean(x[train_idx]))
    if not math.isfinite(mean):
        raise ValueError("Training imputation mean is not finite")
    x[np.isnan(x)] = mean
    if config["normalize"]:
        config["normalization_mean"] = x[train_idx].mean(axis=(0, 1, 3), dtype=np.float64).tolist()
        config["normalization_std"] = np.maximum(
            x[train_idx].std(axis=(0, 1, 3), dtype=np.float64), 1e-8).tolist()
        x = normalize_data(x, config)
    np.savez(output / "split.npz", train_indices=train_idx, validation_indices=val_idx)
    config.update(imputation_mean=mean, train_samples=len(train_idx), validation_samples=len(val_idx),
                  augmentation_seed=config["seed"], augmentation_shift=50,
                  augmentation_noise_fraction=0.05, augmentation_scale=[0.9, 1.1],
                  augmentation_std="per trial/channel temporal population std before augmentation",
                  selection="fixed_epochs_full_train" if config["full_train"] else "validation_accuracy_then_loss",
                  dataset_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(DATA_DIR.glob("*_train.npz"))},
                  environment={"python": platform.python_version(), "torch": str(torch.__version__),
                               "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                               "cuda": torch.version.cuda, "device": str(device),
                               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None})
    write_json(output / "config.json", config)
    train_loader = DataLoader(BCIDataset(x[train_idx], y[train_idx]), batch_size=config["batch_size"],
                              shuffle=True, generator=torch.Generator().manual_seed(config["seed"]))
    val_loader = DataLoader(BCIDataset(x[val_idx], y[val_idx]), batch_size=config["batch_size"],
                            generator=torch.Generator().manual_seed(config["seed"]))
    model = make_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_cls = torch.optim.Adam if config["optimizer"] == "adam" else torch.optim.AdamW
    optimizer = optimizer_cls(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
                 if config["scheduler"] == "cosine" else None)
    generator = torch.Generator(device=device).manual_seed(config["seed"]) if config["augment"] else None
    best_acc, best_loss, best_epoch = -1.0, float("inf"), 0
    history = []
    started = time.perf_counter()
    with (output / "train.log").open("w", encoding="utf-8", buffering=1) as log, \
            (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"]
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for epoch in range(1, config["num_epochs"] + 1):
            lr = optimizer.param_groups[0]["lr"]
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, generator)
            val_loss, val_acc = (None, None) if config["full_train"] else run_epoch(model, val_loader, criterion, device)
            row = dict(zip(fields, [epoch, lr, tr_loss, tr_acc, val_loss, val_acc]))
            history.append(row)
            writer.writerow(row)
            handle.flush()
            selected = (epoch == config["num_epochs"] if config["full_train"] else
                        better_checkpoint(val_acc, val_loss, best_acc, best_loss))
            if selected:
                best_acc, best_loss, best_epoch = val_acc, val_loss, epoch
                weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"state_dict": weights, "config": config, "epoch": epoch,
                            "val_acc": val_acc, "val_loss": val_loss}, output / "best.pt")
            if scheduler:
                scheduler.step()
            line = f'{config["name"]} epoch={epoch:03d} train_acc={tr_acc:.4f}% val_acc={val_acc} val_loss={val_loss}'
            log.write(line + "\n")
            if epoch == 1 or epoch % 50 == 0 or epoch == config["num_epochs"]:
                print(line, flush=True)
    check_loader = train_loader if config["full_train"] else val_loader
    if config["full_train"]:
        check_loader = DataLoader(train_loader.dataset, batch_size=config["batch_size"])
        expected_loss, expected_acc = run_epoch(model, check_loader, criterion, device)
    else:
        expected_loss, expected_acc = best_loss, best_acc
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    actual_loss, actual_acc = run_epoch(model, check_loader, criterion, device)
    if abs(actual_acc - expected_acc) > 1e-8 or abs(actual_loss - expected_loss) > 1e-5:
        raise AssertionError("Reloaded checkpoint does not reproduce recorded evaluation")
    for metric, ylabel in (("acc", "Accuracy (%)"), ("loss", "Cross-entropy loss")):
        splits = ("train",) if config["full_train"] else ("train", "val")
        plot_series({split.title(): [r[f"{split}_{metric}"] for r in history]
                     for split in splits}, ylabel, output / f"{metric}.png")
    result = {"name": config["name"], "model": config["model"], "activation": config["activation"],
              "alpha": config["alpha"], "dropout": config["dropout"], "seed": config["seed"],
              "best_epoch": best_epoch, "val_acc": best_acc, "val_loss": best_loss,
              "selection": config["selection"], "reload_verified": True,
              "parameters": sum(p.numel() for p in model.parameters()),
              "seconds": time.perf_counter() - started}
    if not config["validation_only"]:
        evaluation = evaluate_checkpoint(output / "best.pt")
        result.update(selected_test_acc=evaluation["test_acc"], selected_test_loss=evaluation["test_loss"])
    write_json(output / "summary.json", result)
    return result


def checkpoint_probabilities(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    set_seed(config["seed"])
    for filename, expected in config["dataset_sha256"].items():
        if hashlib.sha256((DATA_DIR / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dataset changed: {filename}")
    _, _, tx, ty = read_bci_data(fill_missing=False)
    tx[np.isnan(tx)] = config["imputation_mean"]
    tx = normalize_data(tx, config)
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    probabilities, losses = [], []
    with torch.no_grad():
        for inputs, labels in DataLoader(BCIDataset(tx, ty), batch_size=config["batch_size"]):
            logits = model(inputs.to(device))
            probabilities.append(logits.softmax(1).cpu())
            losses.append(nn.functional.cross_entropy(logits, labels.to(device), reduction="sum").item())
    return torch.cat(probabilities).numpy(), ty, sum(losses) / len(ty), checkpoint["epoch"]


def evaluate_checkpoint(path):
    probabilities, labels, loss, epoch = checkpoint_probabilities(path)
    result = {"checkpoint": str(path.resolve()), "epoch": epoch,
              "test_loss": loss, "test_acc": float(100 * (probabilities.argmax(1) == labels).mean()),
              "test_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(DATA_DIR.glob("*_test.npz"))}}
    print(json.dumps(result, indent=2), flush=True)
    return result


def evaluate_ensemble(paths):
    predictions = [checkpoint_probabilities(path) for path in paths]
    labels = predictions[0][1]
    assert all(np.array_equal(labels, item[1]) for item in predictions)
    probabilities = np.mean([item[0] for item in predictions], axis=0)
    return {"checkpoints": [str(p.resolve()) for p in paths], "selection": "equal_weight_softmax",
            "test_acc": float(100 * (probabilities.argmax(1) == labels).mean()),
            "test_loss": float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-30)).mean())}


def aggregate_results(results):
    aggregate = {"seeds": [r["seed"] for r in results], "std_ddof": 1}
    for key in ("val_acc", "val_loss", "selected_test_acc", "selected_test_loss"):
        values = [r[key] for r in results if r.get(key) is not None]
        if values:
            aggregate[key + "_mean"] = float(np.mean(values))
            aggregate[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return aggregate


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
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--separable", choices=["depthwise", "full"], default="depthwise")
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--full-train", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results_v2")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--evaluate", type=Path, metavar="CHECKPOINT")
    args = parser.parse_args()
    if args.num_epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, batch size, and finite learning rate must be positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight decay must be finite and nonnegative")
    if args.ensemble and (not args.seeds or args.validation_only):
        parser.error("--ensemble requires --seeds and final test evaluation")
    if args.seeds and len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if not all(0 <= seed < 2**32 for seed in [args.seed, args.split_seed] + (args.seeds or [])):
        parser.error("seed must be in [0, 2**32)")
    if args.evaluate:
        evaluate_checkpoint(args.evaluate)
        return
    base = {k: getattr(args, k) for k in ("num_epochs", "batch_size", "lr", "seed", *DEFAULTS)}
    base.update(seeds=args.seeds or [args.seed], ensemble=args.ensemble)
    if args.suite:
        experiments = EXPERIMENTS
    else:
        dropout = args.dropout if args.dropout is not None else (0.5 if args.model == "deepconvnet" else 0.25)
        experiments = [("single", args.model, args.activation, args.alpha, dropout)]
    if args.seeds:
        experiments = [(f"{name}_seed{seed}", model, activation, alpha, dropout, seed)
                       for name, model, activation, alpha, dropout in experiments for seed in args.seeds]
    else:
        experiments = [(*experiment, args.seed) for experiment in experiments]
    if (args.output / "comparison.csv").exists() or any((args.output / e[0]).exists() for e in experiments):
        parser.error("Experiment directory exists; choose a new --output to preserve earlier results")
    results = []
    for name, model, activation, alpha, dropout, seed in experiments:
        config = dict(base, name=name, model=model, activation=activation, alpha=alpha, dropout=dropout, seed=seed)
        results.append(run_experiment(config, args.output / name))
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    if args.seeds:
        for name, *_ in (EXPERIMENTS if args.suite else [("single",)]):
            group = [r for r in results if r["name"].startswith(name + "_seed")]
            aggregate = aggregate_results(group)
            if args.ensemble:
                aggregate["ensemble"] = evaluate_ensemble([args.output / r["name"] / "best.pt" for r in group])
            write_json(args.output / f"{name}_aggregate.json", aggregate)


if __name__ == "__main__":
    main()
