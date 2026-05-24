import json
import math
import os
import sys

from .agent.run import run
from .utils.generate_summary import generate_summary
from .utils.load_dataset import load_dataset
from .utils.parse_args import parse_args
from .utils.test_model import test_model


def main():
    # Strip --test-only before parse_args (parse_args has its own argparse).
    test_only = "--test-only" in sys.argv
    if test_only:
        sys.argv = [a for a in sys.argv if a != "--test-only"]

    # 1. parse_args() merges yaml + CLI -> args
    args = parse_args()

    # In --test-only, snapshot existing stats.json BEFORE load_dataset overwrites it.
    saved_stats = None
    if test_only:
        stats_path = os.path.join(args.output_dir, "stats.json")
        if not os.path.exists(stats_path):
            print(f"[Main] --test-only requires existing {stats_path}; aborting.")
            return
        with open(stats_path, "r", encoding="utf-8") as f:
            saved_stats = json.load(f)
        if (
            not saved_stats.get("evolution", {}).get("history")
            and saved_stats.get("evolution", {}).get("best_global") is None
        ):
            print(f"[Main] --test-only: {stats_path} has no evolution data; aborting.")
            return

    # 2. load_dataset() loads and preprocesses data -> dataset_dict, stats
    dataset_dict, stats = load_dataset(args)

    if test_only:
        stats = saved_stats
        print("[Main] --test-only: using evolution data from prior stats.json")
    else:
        # 3. run() executes the procedural evolution loop -> updated stats (including best_global)
        try:
            stats = run(args, dataset_dict, stats)
        except Exception as exc:
            print(f"[Error] Evolution failed with exception: {exc}")
            return

    # 4. test_model() final testing and visualization -> train metrics + test plots
    stats = test_model(args, stats, dataset_dict)

    # 5. Generate final summary.json
    summary = generate_summary(stats)

    # 5a. Augment summary with per-member aggregated test metrics across conditions.
    ensemble_info = stats.get("final_test_ensemble", {})
    members_meta = ensemble_info.get("members", [])
    final_test = stats.get("final_test", {})
    per_member_aggregated = []
    for member in members_meta:
        idx = member.get("member_idx")
        test_mses: list[float] = []
        test_maes: list[float] = []
        test_mapes: list[float] = []
        n_ok = 0
        for cond_data in final_test.values():
            for rec in cond_data.get("ensemble_members_per_condition", []):
                if rec.get("member_idx") != idx or rec.get("status") != "ok":
                    continue
                n_ok += 1
                if rec.get("test_mse") is not None:
                    test_mses.append(rec["test_mse"])
                if rec.get("test_mae") is not None:
                    test_maes.append(rec["test_mae"])
                mape = rec.get("test_mape")
                if mape is not None and not math.isnan(mape):
                    test_mapes.append(mape)
        per_member_aggregated.append(
            {
                "member_idx": idx,
                "class_name": member.get("class_name"),
                "source": member.get("source"),
                "val_loss": member.get("val_loss"),
                "n_conditions_ok": n_ok,
                "n_conditions_total": len(final_test),
                "mean_test_mse": (sum(test_mses) / len(test_mses))
                if test_mses
                else None,
                "mean_test_mae": (sum(test_maes) / len(test_maes))
                if test_maes
                else None,
                "mean_test_mape": (sum(test_mapes) / len(test_mapes))
                if test_mapes
                else None,
            }
        )
    summary["ensemble_members"] = per_member_aggregated

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"[Main] Summary saved to {summary_path}")

    print("Done.")


if __name__ == "__main__":
    main()
