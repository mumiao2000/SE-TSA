"""Procedural evolution loop core."""

import argparse
import json
import logging
import os
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from ..utils.evaluate_model import evaluate_model
from .client import LLMClient
from .generate import generate_model_library
from .reflect import reflect

logger = logging.getLogger(__name__)


def run(
    args: argparse.Namespace,
    dataset_dict: dict,
    stats: dict,
) -> dict:
    """Main controller for the procedural evolution loop."""
    # Initialize LLMClient and BertModel once at the start
    llm_client = LLMClient()
    bert_model = SentenceTransformer(args.bert_model_name)

    # Initialize evolution fields
    if "data" not in stats:
        train_stats = stats.get("train_set_stats", {})
        stats["data"] = {
            "n_features": train_stats.get("n_features"),
            "n_train_samples": train_stats.get("n_samples"),
            "column_names": train_stats.get("column_names", []),
            "feature_means": train_stats.get("feature_means"),
            "feature_stds": train_stats.get("feature_stds"),
            "feature_mins": train_stats.get("feature_mins"),
            "feature_maxs": train_stats.get("feature_maxs"),
            "trend_strengths": train_stats.get("trend_strengths"),
            "dominant_period": train_stats.get("dominant_period"),
        }
    if "evolution" not in stats:
        stats["evolution"] = {}
    stats["evolution"].setdefault("current_iteration", 0)
    stats["evolution"].setdefault("best_global", None)
    stats["evolution"].setdefault("history", [])
    stats["evolution"].setdefault("failures", [])
    stats["evolution"].setdefault("principles", [])
    stats["evolution"].setdefault("exploration_rate", args.exploration_rate_base)
    stats["evolution"].setdefault(
        "map_elites",
        _init_map_elites_grid(args.arch_families, args.preprocessing_types),
    )
    stats["evolution"].setdefault(
        "time_tracking",
        {
            "iteration_start_times": [],
            "iteration_durations": [],
            "total_elapsed_minutes": 0.0,
        },
    )

    if "experiment_meta" not in stats:
        stats["experiment_meta"] = {}
    stats["experiment_meta"]["start_time"] = time.time()
    stats["experiment_meta"]["status"] = "running"
    stats["experiment_meta"]["max_time"] = args.max_time
    stats["experiment_meta"]["max_iterations"] = args.max_iterations

    base_eval_cfg = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_max_epochs": args.train_max_epochs,
        "train_patience": args.train_patience,
        "device": "auto",
        "quick_search_max_epochs": args.quick_search_max_epochs,
        "quick_search_patience": args.quick_search_patience,
        "quick_search_max_time": args.quick_search_max_time,
        "quick_search_max_combinations": args.quick_search_max_combinations,
        "mape_threshold": args.mape_threshold,
    }

    start_time = time.time()

    while True:
        iteration = stats["evolution"]["current_iteration"]
        elapsed = (time.time() - start_time) / 60.0
        stats["evolution"]["time_tracking"]["total_elapsed_minutes"] = elapsed

        # Stopping conditions
        if elapsed >= args.max_time:
            print(f"[Core] Time budget exceeded ({elapsed:.1f} min). Stopping.")
            break
        if iteration >= args.max_iterations:
            print(f"[Core] Max iterations ({args.max_iterations}) reached. Stopping.")
            break

        print(f"\n[Core] === Iteration {iteration} ===")
        stats["evolution"]["time_tracking"]["iteration_start_times"].append(time.time())

        # Generate model library
        candidates = generate_model_library(args, stats, llm_client)
        if not candidates:
            print("[Core] No candidates generated. Skipping.")
            stats["evolution"]["current_iteration"] += 1
            continue

        # Evaluate each candidate across all conditions
        for cand in candidates:
            class_name = cand["class_name"]
            model_path = cand["model_path"]

            per_condition_results = {}
            per_condition_configs = {}
            all_success = True

            for (il, ol), (train_xy, val_xy, _) in dataset_dict.items():
                condition_cfg = {**base_eval_cfg, "input_len": il, "output_len": ol}
                result = evaluate_model(
                    model_path=model_path,
                    class_name=class_name,
                    config=None,
                    train_xy=train_xy,
                    eval_xy=val_xy,
                    eval_cfg=condition_cfg,
                )
                if result["status"] != "success":
                    all_success = False
                    print(
                        f"[Core] {class_name} [{il}x{ol}] FAILED: {result['status']} – {result['error_message']}"
                    )
                    break
                key = f"{il}_{ol}"
                per_condition_results[key] = result
                per_condition_configs[key] = result["best_config"]
                print(
                    f"[Core] {class_name} [{il}x{ol}] OK: mse={result['mse']:.6f} mae={result['mae']:.6f} "
                    f"params={result['num_params']} epochs={result['epochs_trained']} time={result['training_time']:.1f}s"
                )

            if all_success:
                mean_mse = float(
                    np.mean([r["mse"] for r in per_condition_results.values()])
                )
                mean_mae = float(
                    np.mean([r["mae"] for r in per_condition_results.values()])
                )
                mean_mape = float(
                    np.mean([r["mape"] for r in per_condition_results.values()])
                )
                mean_num_params = float(
                    np.mean([r["num_params"] for r in per_condition_results.values()])
                )
                mean_training_time = float(
                    np.mean(
                        [r["training_time"] for r in per_condition_results.values()]
                    )
                )
                mean_epochs_trained = float(
                    np.mean(
                        [r["epochs_trained"] for r in per_condition_results.values()]
                    )
                )

                entry = {
                    "iteration": iteration,
                    "class_name": class_name,
                    "val_loss": mean_mse,
                    "mae": mean_mae,
                    "mape": mean_mape,
                    "num_params": mean_num_params,
                    "arch_family": cand.get("arch_family", "hybrid"),
                    "preprocessing": cand.get("preprocessing", "hybrid"),
                    "per_condition_results": {
                        k: {"mse": v["mse"], "mae": v["mae"], "mape": v["mape"]}
                        for k, v in per_condition_results.items()
                    },
                    "per_condition_configs": per_condition_configs,
                    "best_config": per_condition_configs[
                        list(per_condition_configs.keys())[0]
                    ],
                    "model_path": model_path,
                    "training_time": mean_training_time,
                    "epochs_trained": mean_epochs_trained,
                    "description": cand.get("description", ""),
                    "design_rationale": cand.get("design_rationale", ""),
                    "timestamp": time.time(),
                }
                stats["evolution"]["history"].append(entry)
                _update_map_elites_cell(stats["evolution"]["map_elites"], entry)

                best_global = stats["evolution"]["best_global"]
                if best_global is None or mean_mse < best_global["val_loss"]:
                    stats["evolution"]["best_global"] = entry
                    print(f"[Core] New best: {class_name} val_loss={mean_mse:.6f}")
                else:
                    print(f"[Core] Evaluated {class_name}: val_loss={mean_mse:.6f}")
            else:
                failure = {
                    "iteration": iteration,
                    "class_name": class_name,
                    "status": result["status"],
                    "error_message": result["error_message"],
                    "timestamp": time.time(),
                }
                stats["evolution"]["failures"].append(failure)
                print(
                    f"[Core] Failed {class_name}: {result['status']} – {result['error_message']}"
                )

        # Reflect and update
        if iteration > 0:
            reflect(args, stats, llm_client, bert_model)

        # Decay exploration_rate
        current_rate = stats["evolution"]["exploration_rate"]
        new_rate = max(
            args.exploration_rate_min, current_rate * args.exploration_rate_decay
        )
        stats["evolution"]["exploration_rate"] = new_rate

        stats["evolution"]["current_iteration"] += 1
        iter_duration = (
            time.time()
            - stats["evolution"]["time_tracking"]["iteration_start_times"][-1]
        ) / 60.0
        stats["evolution"]["time_tracking"]["iteration_durations"].append(iter_duration)

        # Real-time persistence
        _save_stats(stats, args.output_dir)

    stats["experiment_meta"]["end_time"] = time.time()
    stats["experiment_meta"]["status"] = "evolution_completed"
    _save_stats(stats, args.output_dir)
    return stats


def _init_map_elites_grid(arch_families, preprocessing_types) -> dict:
    """Initialize the MAP-Elites grid structure."""
    grid = {
        "grid_shape": ["arch_family", "preprocessing"],
        "arch_families": arch_families,
        "preprocessing_types": preprocessing_types,
        "cells": {},
        "coverage": 0.0,
        "pareto_front": [],
    }
    for af in arch_families:
        for pt in preprocessing_types:
            key = f"{af}|{pt}"
            grid["cells"][key] = {
                "occupied": False,
                "best_val_loss": None,
                "best_entry": None,
                "all_entries": [],
            }
    return grid


def _update_map_elites_cell(grid: dict, entry: dict) -> None:
    """Update the corresponding cell in the MAP-Elites grid."""
    af = entry.get("arch_family", "hybrid")
    pp = entry.get("preprocessing", "hybrid")
    key = f"{af}|{pp}"
    if key not in grid["cells"]:
        snap_af = af if af in grid["arch_families"] else "hybrid"
        snap_pp = pp if pp in grid["preprocessing_types"] else "hybrid"
        new_key = f"{snap_af}|{snap_pp}"
        logger.warning(
            "MAP-Elites snap: '%s' -> '%s' (af=%s pp=%s)",
            key,
            new_key,
            af,
            pp,
        )
        if new_key not in grid["cells"]:
            logger.warning("MAP-Elites drop: '%s' still invalid after snap", new_key)
            return
        entry["arch_family"] = snap_af
        entry["preprocessing"] = snap_pp
        key = new_key

    cell = grid["cells"][key]
    cell["occupied"] = True
    cell["all_entries"].append(entry)

    if cell["best_val_loss"] is None or entry["val_loss"] < cell["best_val_loss"]:
        cell["best_val_loss"] = entry["val_loss"]
        cell["best_entry"] = entry

    # Recompute coverage
    total_cells = len(grid["cells"])
    occupied_cells = sum(1 for c in grid["cells"].values() if c["occupied"])
    grid["coverage"] = occupied_cells / total_cells if total_cells > 0 else 0.0

    # Recompute pareto_front
    all_entries = []
    for c in grid["cells"].values():
        if c["best_entry"] is not None:
            all_entries.append(c["best_entry"])

    pareto = []
    for e in all_entries:
        dominated = False
        for other in all_entries:
            if other is e:
                continue
            if (
                other["num_params"] <= e["num_params"]
                and other["val_loss"] <= e["val_loss"]
            ):
                if (
                    other["num_params"] < e["num_params"]
                    or other["val_loss"] < e["val_loss"]
                ):
                    dominated = True
                    break
        if not dominated:
            pareto.append(
                {
                    "class_name": e["class_name"],
                    "val_loss": e["val_loss"],
                    "num_params": e["num_params"],
                }
            )
    grid["pareto_front"] = sorted(pareto, key=lambda x: x["num_params"])


def _save_stats(stats, output_dir):
    """Write stats to output_dir/stats.json."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "stats.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
