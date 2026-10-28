"""Audit saved selections, splits, aggregates and preservation without new test access."""
import csv
import json

import numpy as np

from experiments_v2 import original_hashes
from main import ROOT


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    root = ROOT / "results_v2"
    freeze = load(root / "frozen_selection.json")
    assert original_hashes() == load(root / "protocol.json")["original_results_sha256"]
    comparison = list(csv.DictReader((root / "comparison.csv").open(encoding="utf-8-sig")))
    reference = np.load(root / "c00/seed42/split.npz")
    total_epochs = 0
    for row in comparison:
        directory = root / row["candidate"] / f'seed{row["seed"]}'
        config, summary = load(directory / "config.json"), load(directory / "summary.json")
        history = list(csv.DictReader((directory / "history.csv").open(encoding="utf-8")))
        total_epochs += len(history)
        assert len(history) == config["num_epochs"]
        assert not any("test" in key for key in history[0])
        assert summary["reload_verified"]
        assert float(row["selected_test_acc"]) == summary["selected_test_acc"]
        split = np.load(directory / "split.npz")
        if config["full_train"]:
            assert len(split["train_indices"]) == 1080 and len(split["validation_indices"]) == 0
            assert summary["selection"] == "fixed_epochs_full_train"
            assert summary["best_epoch"] == freeze["full_train_epochs"] == len(history)
        else:
            assert all(np.array_equal(split[k], reference[k]) for k in reference.files)
            best = min(history, key=lambda r: (-float(r["val_acc"]), float(r["val_loss"]), int(r["epoch"])))
            assert int(best["epoch"]) == summary["best_epoch"]
            assert float(best["val_acc"]) == summary["val_acc"]
            assert float(best["val_loss"]) == summary["val_loss"]
        for filename in ("acc.png", "loss.png", "best.pt", "evaluation.json"):
            assert (directory / filename).stat().st_size > 0
    candidates = load(root / "candidates.json")
    for candidate in candidates:
        assert len(candidate["runs"]) == 3
        assert np.isclose(np.mean([r["val_acc"] for r in candidate["runs"]]), candidate["aggregate"]["val_acc_mean"])
    by_name = {c["candidate"]: c for c in candidates}
    for stage in load(root / "decisions.json"):
        selected = min([by_name[c] for c in stage["candidates"]],
                       key=lambda c: (-c["aggregate"]["val_acc_mean"], c["aggregate"]["val_loss_mean"]))
        assert stage["selected"] == selected["candidate"]
    final = load(root / "final_summary.json")
    for name, label in ((freeze["candidate"], "validation_selected"), ("full_train", "full_train")):
        values = [float(r["selected_test_acc"]) for r in comparison if r["candidate"] == name]
        assert len(values) == 5
        assert np.isclose(np.mean(values), final[label]["selected_test_acc_mean"])
        assert np.isclose(np.std(values, ddof=1), final[label]["selected_test_acc_std"])
    assert len(comparison) == len(candidates) * 3 + 2 + 5
    print(f"PASS: {len(candidates)} candidates, {len(comparison)} runs, {total_epochs} epochs; "
          "validation-only selection, fixed splits, summaries, curves, aggregates and original results preserved")


if __name__ == "__main__":
    main()
