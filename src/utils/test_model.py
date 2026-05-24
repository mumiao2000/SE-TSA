"""Final test evaluation and visualization with bagging ensemble.

Note: ``_compute_metrics_from_arrays`` is a numpy mirror of the torch math at
the tail of ``evaluate_model.eval_metrics``. If that function changes (e.g.
weighted MAPE), this helper must be updated in lockstep — they are intentionally
parallel implementations because the constraint is to leave evaluate_model.py
untouched.
"""

import argparse
import importlib.util
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .evaluate_model import eval_metrics, train_one_model

logger = logging.getLogger(__name__)


def _compute_metrics_from_arrays(
    pred: np.ndarray, target: np.ndarray, mape_threshold: float
) -> tuple[float, float, float]:
    """MSE/MAE/MAPE on column 0; mirrors evaluate_model.eval_metrics tail."""
    pred_c0 = pred[..., 0]
    target_c0 = target[..., 0]
    mse = float(np.mean((pred_c0 - target_c0) ** 2))
    mae = float(np.mean(np.abs(pred_c0 - target_c0)))
    mask = np.abs(target_c0) > mape_threshold
    if mask.sum() > 0:
        mape = float(
            np.mean(np.abs((pred_c0[mask] - target_c0[mask]) / target_c0[mask])) * 100
        )
    else:
        mape = float("nan")
    return mse, mae, mape


def _select_ensemble_members(
    stats: dict, n_overall: int, n_map_elites: int
) -> list[tuple[dict, str]]:
    """Top-N from history (overall) + Top-N from MAP-Elites cell bests.

    Duplicates between the two pools are kept on purpose: a model appearing in
    both lists effectively gets double weight in the average.
    """
    history = stats.get("evolution", {}).get("history", [])
    top_overall = sorted(history, key=lambda e: e.get("val_loss", float("inf")))[
        :n_overall
    ]

    cells = stats.get("evolution", {}).get("map_elites", {}).get("cells", {})
    cell_bests = [
        c["best_entry"]
        for c in cells.values()
        if c.get("occupied") and c.get("best_entry") is not None
    ]
    top_map = sorted(cell_bests, key=lambda e: e.get("val_loss", float("inf")))[
        :n_map_elites
    ]

    members: list[tuple[dict, str]] = [(e, "overall") for e in top_overall]
    members += [(e, "map_elites") for e in top_map]

    if not members:
        best = stats.get("evolution", {}).get("best_global")
        if best is None:
            raise RuntimeError(
                "No ensemble members available "
                "(history/map_elites/best_global all empty)"
            )
        members = [(best, "best_global")]
    return members


def _train_and_predict_member(
    member: dict,
    member_idx: int,
    il: int,
    ol: int,
    train_xy,
    val_xy,
    test_xy,
    args: argparse.Namespace,
    device: torch.device,
) -> dict | None:
    """Retrain a single ensemble member and return its train/test predictions.

    Returns None on any failure (after logging a warning) so the caller can
    skip and continue with remaining members.
    """
    class_name = member["class_name"]
    model_path = member["model_path"]
    spec_name = f"_final_{member_idx}_{class_name}"

    try:
        spec = importlib.util.spec_from_file_location(spec_name, model_path)
        if spec is None or spec.loader is None:
            logger.warning(
                "Member %d (%s) skipped: cannot load spec from %s",
                member_idx,
                class_name,
                model_path,
            )
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec_name] = mod
        spec.loader.exec_module(mod)
        ModelCls = getattr(mod, class_name)

        cond_key = f"{il}_{ol}"
        config = member.get("per_condition_configs", {}).get(
            cond_key, member.get("best_config")
        )
        n_features = train_xy[0].shape[2]

        if config is not None:
            model = ModelCls(il, ol, n_features, config).to(device)
        else:
            model = ModelCls(il, ol, n_features).to(device)

        batch_size = args.batch_size
        pin_memory = device.type == "cuda"

        train_dataset = TensorDataset(
            torch.from_numpy(train_xy[0]).float(),
            torch.from_numpy(train_xy[1]).float(),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(val_xy[0]).float(),
            torch.from_numpy(val_xy[1]).float(),
        )
        test_dataset = TensorDataset(
            torch.from_numpy(test_xy[0]).float(),
            torch.from_numpy(test_xy[1]).float(),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
        )
        # Separate non-shuffled loader so train predictions align across members
        train_eval_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )

        train_one_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=args.train_max_epochs,
            lr=args.lr,
            patience=args.train_patience,
        )

        train_mse, train_mae, train_mape, train_pred, train_target = eval_metrics(
            model,
            train_eval_loader,
            device,
            args.mape_threshold,
            return_predictions=True,
        )
        test_mse, test_mae, test_mape, test_pred, test_target = eval_metrics(
            model,
            test_loader,
            device,
            args.mape_threshold,
            return_predictions=True,
        )

        out = {
            "train_pred": train_pred,
            "train_target": train_target,
            "test_pred": test_pred,
            "test_target": test_target,
            "class_name": class_name,
            "val_loss": member.get("val_loss"),
            "train_mse": train_mse,
            "train_mae": train_mae,
            "train_mape": train_mape,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_mape": test_mape,
        }

        del model, train_loader, train_eval_loader, val_loader, test_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return out

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Member %d (%s) failed at [%dx%d]: %s",
            member_idx,
            class_name,
            il,
            ol,
            exc,
        )
        return None
    finally:
        if spec_name in sys.modules:
            del sys.modules[spec_name]


def test_model(
    args: argparse.Namespace,
    stats: dict,
    dataset_dict: dict,
) -> dict:
    """Bagging ensemble: top-N overall + top-N MAP-Elites, equal-weight mean."""
    members = _select_ensemble_members(
        stats,
        args.ensemble_top_n_overall,
        args.ensemble_top_n_map_elites,
    )

    device_str = getattr(args, "device", "auto")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and device_str == "auto"
        else (device_str if device_str != "auto" else "cpu")
    )

    members_meta = [
        {
            "member_idx": idx,
            "class_name": m["class_name"],
            "model_path": m["model_path"],
            "source": src,
            "val_loss": m.get("val_loss"),
        }
        for idx, (m, src) in enumerate(members)
    ]
    print(
        f"[Test] Ensemble = {len(members)} members "
        f"(top-{args.ensemble_top_n_overall} overall + "
        f"top-{args.ensemble_top_n_map_elites} map-elites)"
    )

    final_test_results: dict = {}

    for (il, ol), (train_xy, val_xy, test_xy) in dataset_dict.items():
        key = f"{il}_{ol}"
        member_outputs: list[dict] = []
        per_member_records: list[dict] = []

        for idx, (m, src) in enumerate(members):
            out = _train_and_predict_member(
                m, idx, il, ol, train_xy, val_xy, test_xy, args, device
            )
            if out is None:
                per_member_records.append(
                    {
                        "member_idx": idx,
                        "class_name": m["class_name"],
                        "source": src,
                        "status": "failed",
                    }
                )
                print(f"[Test] {key} member#{idx} ({m['class_name']}, {src}): FAILED")
                continue
            per_member_records.append(
                {
                    "member_idx": idx,
                    "class_name": m["class_name"],
                    "source": src,
                    "status": "ok",
                    "val_loss": m.get("val_loss"),
                    "train_mse": out["train_mse"],
                    "train_mae": out["train_mae"],
                    "train_mape": out["train_mape"],
                    "test_mse": out["test_mse"],
                    "test_mae": out["test_mae"],
                    "test_mape": out["test_mape"],
                }
            )
            member_outputs.append(out)
            print(
                f"[Test] {key} member#{idx} ({m['class_name']}, {src}): "
                f"test_mse={out['test_mse']:.6f} test_mae={out['test_mae']:.6f} "
                f"train_mse={out['train_mse']:.6f}"
            )

        if not member_outputs:
            logger.error("All ensemble members failed for %s", key)
            final_test_results[key] = {
                "train_mse": None,
                "train_mae": None,
                "train_mape": None,
                "test_mse": None,
                "test_mae": None,
                "test_mape": None,
                "test_plot_path": None,
                "n_members_used": 0,
                "n_members_expected": len(members),
                "ensemble_members_per_condition": per_member_records,
            }
            continue

        if len(member_outputs) < len(members):
            logger.warning(
                "Ensemble degraded for %s: %d/%d members succeeded",
                key,
                len(member_outputs),
                len(members),
            )

        train_mean_pred = np.mean(
            np.stack([m["train_pred"] for m in member_outputs]), axis=0
        )
        train_target = member_outputs[0]["train_target"]
        test_mean_pred = np.mean(
            np.stack([m["test_pred"] for m in member_outputs]), axis=0
        )
        test_target = member_outputs[0]["test_target"]

        train_mse, train_mae, train_mape = _compute_metrics_from_arrays(
            train_mean_pred, train_target, args.mape_threshold
        )
        test_mse, test_mae, test_mape = _compute_metrics_from_arrays(
            test_mean_pred, test_target, args.mape_threshold
        )

        plot_path = os.path.join(args.output_dir, f"test_prediction_{key}.png")
        _plot_first_window(
            test_xy[0],
            test_mean_pred,
            test_target,
            il,
            ol,
            plot_path,
            f"Ensemble_{len(member_outputs)}",
        )

        final_test_results[key] = {
            "train_mse": train_mse,
            "train_mae": train_mae,
            "train_mape": train_mape,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_mape": test_mape,
            "test_plot_path": plot_path,
            "n_members_used": len(member_outputs),
            "n_members_expected": len(members),
            "ensemble_members_per_condition": per_member_records,
        }
        print(
            f"[Test] {key}: members={len(member_outputs)}/{len(members)} "
            f"test_mse={test_mse:.6f} test_mae={test_mae:.6f}"
        )

    stats["final_test"] = final_test_results
    stats["final_test_ensemble"] = {"members": members_meta}

    import json

    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    return stats


def _plot_first_window(
    inputs, predictions, targets, input_len, output_len, plot_path, class_name
):
    """Draw a prediction visualization for the first window of the test set."""
    input_seq = inputs[0, :, 0]
    true_future = targets[0, :, 0]
    pred_future = predictions[0, :, 0]

    x_input = list(range(input_len))
    x_output = list(range(input_len, input_len + output_len))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axvspan(
        input_len - 0.5,
        input_len + output_len - 0.5,
        color="#f0f0f0",
        alpha=0.6,
        zorder=0,
    )
    ax.plot(
        x_input,
        input_seq,
        color="#2c3e50",
        linewidth=2.2,
        label="Historical Input",
        zorder=3,
    )
    ax.plot(
        x_output,
        true_future,
        color="#2980b9",
        linewidth=2.2,
        label="Ground Truth",
        zorder=3,
    )
    ax.plot(
        x_output,
        pred_future,
        color="#e74c3c",
        linewidth=2.2,
        linestyle="--",
        label="Prediction",
        zorder=3,
    )
    ax.plot(
        [x_input[-1], x_output[0]],
        [input_seq[-1], true_future[0]],
        color="#2980b9",
        linewidth=1.5,
        alpha=0.4,
        zorder=2,
    )
    ax.plot(
        [x_input[-1], x_output[0]],
        [input_seq[-1], pred_future[0]],
        color="#e74c3c",
        linewidth=1.5,
        alpha=0.4,
        linestyle="--",
        zorder=2,
    )
    ax.axvline(
        x=input_len - 0.5,
        color="#7f8c8d",
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
        zorder=1,
    )
    ax.text(
        input_len - 0.5 + 0.15,
        ax.get_ylim()[1] * 0.98 + ax.get_ylim()[0] * 0.02,
        "Forecast →",
        color="#7f8c8d",
        fontsize=10,
        va="top",
    )
    ax.set_title(
        f"Test Prediction — {class_name}",
        fontsize=14,
        fontweight="bold",
        color="#2c3e50",
    )
    ax.set_xlabel("Time Step", fontsize=12, color="#34495e")
    ax.set_ylabel("Target Value", fontsize=12, color="#34495e")
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
