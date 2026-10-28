"""Re-evaluate the frozen V2 checkpoints and print the final test accuracies."""
import json

from main import ROOT, checkpoint_probabilities, evaluate_ensemble


def main():
    root = ROOT / "results_v2"
    freeze = json.loads((root / "frozen_selection.json").read_text(encoding="utf-8"))
    print(f'frozen candidate: {freeze["candidate"]}  seeds: {freeze["seeds"]}')
    for name, label in ((freeze["candidate"], "864-train, validation-selected"),
                        ("full_train", f'1080-train, fixed {freeze["full_train_epochs"]} epochs')):
        paths = [root / name / f"seed{s}" / "best.pt" for s in freeze["seeds"]]
        print(f"\n[{label}]")
        for seed, path in zip(freeze["seeds"], paths):
            probabilities, labels, loss, epoch = checkpoint_probabilities(path)
            acc = 100 * (probabilities.argmax(1) == labels).mean()
            print(f"  seed {seed}  epoch {epoch:3d}  test_acc {acc:.4f}%  test_loss {loss:.4f}")
        result = evaluate_ensemble(paths)
        print(f'  5-seed softmax ensemble  test_acc {result["test_acc"]:.4f}%  test_loss {result["test_loss"]:.4f}')


if __name__ == "__main__":
    main()
