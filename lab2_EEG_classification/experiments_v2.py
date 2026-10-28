"""Validation-only greedy search, followed by a separate frozen final evaluation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from main import (ROOT, DEFAULTS, aggregate_results, evaluate_checkpoint, evaluate_ensemble,
                  run_experiment, write_json)


STAGES = [
    ("separable", [{"separable": v} for v in ("depthwise", "full")]),
    ("weight_decay", [{"weight_decay": v} for v in (0.0, 1e-4, 1e-3)]),
    ("lr_scheduler", [{"lr": lr, "scheduler": scheduler}
                      for lr in (0.01, 0.001) for scheduler in ("none", "cosine")]),
    ("normalize", [{"normalize": v} for v in (False, True)]),
    ("augment", [{"augment": v} for v in (False, True)]),
    ("dropout", [{"dropout": v} for v in (0.1, 0.25, 0.5)]),
    ("activation", [{"activation": "relu", "alpha": 1.0},
                    {"activation": "leakyrelu", "alpha": 1.0},
                    {"activation": "elu", "alpha": 0.5}, {"activation": "elu", "alpha": 1.0}]),
    ("optimizer", [{"optimizer": v} for v in ("adam", "adamw")]),
    ("epochs", [{"num_epochs": v} for v in (150, 300)]),
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def original_hashes():
    return {str(p.relative_to(ROOT / "results")): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "results").rglob("*")) if p.is_file()}


def search(output):
    output.mkdir(parents=True, exist_ok=False)
    base = DEFAULTS | dict(model="eegnet", activation="relu", alpha=1.0, dropout=0.25,
                           num_epochs=150, batch_size=64, lr=0.01, validation_only=True)
    write_json(output / "protocol.json", dict(
        base=base, stages=STAGES, search_seeds=[42, 43, 44], final_seeds=[42, 43, 44, 45, 46],
        selection="mean validation accuracy descending, mean validation loss ascending; earlier candidate on exact ties",
        full_train_epochs="median of five validation-selected epochs, rounded half up",
        primary_endpoint="five-seed full-train equal-weight softmax ensemble",
        test_policy="no test predictions until all configurations, seeds and full-train epochs are frozen; no further tuning",
        std_ddof=1, original_results_sha256=original_hashes()))
    candidates, cache, decisions = [], {}, []

    def candidate(config):
        key = json.dumps(config, sort_keys=True)
        if key not in cache:
            name = f"c{len(candidates):02d}"
            runs = [run_experiment(config | dict(name=name, seed=seed), output / name / f"seed{seed}")
                    for seed in (42, 43, 44)]
            item = dict(candidate=name, config=config, aggregate=aggregate_results(runs), runs=runs)
            candidates.append(item)
            cache[key] = item
            write_json(output / "candidates.json", candidates)
        return cache[key]

    current = candidate(base)
    for stage, options in STAGES:
        before = current
        choices = [candidate(current["config"] | option) for option in options]
        current = min(choices, key=lambda c: (-c["aggregate"]["val_acc_mean"], c["aggregate"]["val_loss_mean"]))
        decisions.append(dict(stage=stage, parent=before["candidate"],
                              candidates=[c["candidate"] for c in choices], selected=current["candidate"]))
        write_json(output / "decisions.json", decisions)
        print(f'STAGE {stage}: {current["candidate"]} validation={current["aggregate"]["val_acc_mean"]:.4f}', flush=True)
    selected = current["candidate"]
    for seed in (45, 46):
        current["runs"].append(run_experiment(current["config"] | dict(name=selected, seed=seed),
                                               output / selected / f"seed{seed}"))
    epochs = int(np.floor(np.median([r["best_epoch"] for r in current["runs"]]) + 0.5))
    freeze = dict(candidate=selected, config=current["config"], seeds=[42, 43, 44, 45, 46],
                  full_train_epochs=epochs, validation_epochs=[r["best_epoch"] for r in current["runs"]],
                  validation_aggregate=aggregate_results(current["runs"]),
                  primary_endpoint="five-seed full-train equal-weight softmax ensemble",
                  candidates_sha256=hashlib.sha256((output / "candidates.json").read_bytes()).hexdigest(),
                  protocol_sha256=hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest(),
                  decisions_sha256=hashlib.sha256((output / "decisions.json").read_bytes()).hexdigest())
    write_json(output / "frozen_selection.json", freeze)
    assert original_hashes() == read_json(output / "protocol.json")["original_results_sha256"]
    print("FROZEN " + json.dumps(freeze, ensure_ascii=False), flush=True)


def final(output):
    freeze = read_json(output / "frozen_selection.json")
    for filename, field in (("candidates.json", "candidates_sha256"), ("protocol.json", "protocol_sha256"),
                            ("decisions.json", "decisions_sha256")):
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == freeze[field]
    if (output / "final_evaluation_started.json").exists():
        raise FileExistsError("Final evaluation already started; do not overwrite or tune after test access")
    full_runs = []
    for seed in freeze["seeds"]:
        config = freeze["config"] | dict(name="full_train", seed=seed, full_train=True,
                                          num_epochs=freeze["full_train_epochs"], validation_only=True)
        full_runs.append(run_experiment(config, output / "full_train" / f"seed{seed}"))
    write_json(output / "final_evaluation_started.json", dict(
        frozen_selection_sha256=hashlib.sha256((output / "frozen_selection.json").read_bytes()).hexdigest(),
        policy="All training completed before this marker; final reporting only"))
    candidates = read_json(output / "candidates.json")
    comparisons, ablations = [], []
    for item in candidates + [dict(candidate="full_train", config=config)]:
        name = item["candidate"]
        seeds = freeze["seeds"] if name in (freeze["candidate"], "full_train") else [42, 43, 44]
        results = []
        for seed in seeds:
            directory = output / name / f"seed{seed}"
            result = read_json(directory / "summary.json")
            evaluation = evaluate_checkpoint(directory / "best.pt")
            write_json(directory / "evaluation.json", evaluation)
            result.update(selected_test_acc=evaluation["test_acc"], selected_test_loss=evaluation["test_loss"])
            write_json(directory / "summary.json", result)
            comparisons.append(item["config"] | result | dict(candidate=name))
            results.append(result)
        aggregate = aggregate_results(results)
        write_json(output / name / "aggregate.json", aggregate)
        # Use the same three seeds for all ablation contrasts.
        ablations.append(dict(candidate=name, **item["config"], **aggregate_results(results[:3])))
    write_csv(output / "comparison.csv", comparisons)
    baseline = ablations[0]
    for row in ablations:
        row["validation_delta_vs_baseline_pp"] = (row.get("val_acc_mean", 0) - baseline["val_acc_mean"]
                                                   if row["candidate"] != "full_train" else None)
        row["test_delta_vs_baseline_pp"] = row["selected_test_acc_mean"] - baseline["selected_test_acc_mean"]
    write_csv(output / "ablation.csv", ablations)
    stage_rows = []
    by_name = {r["candidate"]: r for r in ablations}
    for stage in read_json(output / "decisions.json"):
        parent = by_name[stage["parent"]]
        for name in stage["candidates"]:
            row = by_name[name]
            stage_rows.append(dict(stage=stage["stage"], parent=stage["parent"], candidate=name,
                                   selected=(name == stage["selected"]),
                                   validation_mean=row["val_acc_mean"], validation_std=row["val_acc_std"],
                                   test_mean=row["selected_test_acc_mean"], test_std=row["selected_test_acc_std"],
                                   validation_delta_pp=row["val_acc_mean"]-parent["val_acc_mean"],
                                   test_delta_pp=row["selected_test_acc_mean"]-parent["selected_test_acc_mean"]))
    write_csv(output / "ablation_stages.csv", stage_rows)
    summary = dict(frozen_selection=freeze)
    for name, label in ((freeze["candidate"], "validation_selected"), ("full_train", "full_train")):
        summary[label] = read_json(output / name / "aggregate.json")
        summary[label]["ensemble"] = evaluate_ensemble([output / name / f"seed{s}" / "best.pt" for s in freeze["seeds"]])
    write_json(output / "final_summary.json", summary)
    assert original_hashes() == read_json(output / "protocol.json")["original_results_sha256"]
    write_json(output / "integrity.json", dict(original_results_unchanged=True, all_runs_reloaded=True,
                                                no_epochwise_test=True, final_selection_frozen=True))
    print("FINAL " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["search", "final"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results_v2")
    args = parser.parse_args()
    (search if args.phase == "search" else final)(args.output)
