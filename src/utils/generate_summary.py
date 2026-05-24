"""Generate simplified summary from full stats."""


def generate_summary(stats: dict) -> dict:
    """
    Extract concise information from full stats to generate summary.json content.
    Includes: best_global, top-10 history, principles, per-condition metrics,
              mean metrics, map_elites.coverage, total_iterations, time_used
    """
    evolution = stats.get("evolution", {})
    best = evolution.get("best_global")
    history = evolution.get("history", [])
    final_test = stats.get("final_test", {})
    time_tracking = evolution.get("time_tracking", {})

    top10 = sorted(history, key=lambda e: e.get("val_loss", float("inf")))[:10]

    summary = {
        "best_model": {
            "class_name": best.get("class_name") if best else None,
            "val_loss": best.get("val_loss") if best else None,
            "num_params": best.get("num_params") if best else None,
            "arch_family": best.get("arch_family") if best else None,
            "preprocessing": best.get("preprocessing") if best else None,
            "best_config": best.get("best_config") if best else None,
            "per_condition_configs": best.get("per_condition_configs")
            if best
            else None,
        },
        "top_10": [
            {
                "class_name": e.get("class_name"),
                "val_loss": e.get("val_loss"),
                "num_params": e.get("num_params"),
                "arch_family": e.get("arch_family"),
                "preprocessing": e.get("preprocessing"),
                "iteration": e.get("iteration"),
            }
            for e in top10
        ],
        "principles": evolution.get("principles", []),
        "per_condition_train_metrics": {},
        "per_condition_test_metrics": {},
        "mean_train_metrics": {},
        "mean_test_metrics": {},
        "map_elites_coverage": evolution.get("map_elites", {}).get("coverage"),
        "total_iterations": evolution.get("current_iteration", 0),
        "time_used_minutes": time_tracking.get("total_elapsed_minutes", 0.0),
    }

    # Fill per-condition metrics
    train_mses = []
    train_maes = []
    train_mapes = []
    test_mses = []
    test_maes = []
    test_mapes = []

    for key, cond in final_test.items():
        summary["per_condition_train_metrics"][key] = {
            "mse": cond.get("train_mse"),
            "mae": cond.get("train_mae"),
            "mape": cond.get("train_mape"),
        }
        summary["per_condition_test_metrics"][key] = {
            "mse": cond.get("test_mse"),
            "mae": cond.get("test_mae"),
            "mape": cond.get("test_mape"),
        }
        if cond.get("train_mse") is not None:
            train_mses.append(cond["train_mse"])
        if cond.get("train_mae") is not None:
            train_maes.append(cond["train_mae"])
        if cond.get("train_mape") is not None:
            train_mapes.append(cond["train_mape"])
        if cond.get("test_mse") is not None:
            test_mses.append(cond["test_mse"])
        if cond.get("test_mae") is not None:
            test_maes.append(cond["test_mae"])
        if cond.get("test_mape") is not None:
            test_mapes.append(cond["test_mape"])

    if train_mses:
        summary["mean_train_metrics"] = {
            "mse": sum(train_mses) / len(train_mses),
            "mae": sum(train_maes) / len(train_maes) if train_maes else None,
            "mape": sum(train_mapes) / len(train_mapes) if train_mapes else None,
        }
    if test_mses:
        summary["mean_test_metrics"] = {
            "mse": sum(test_mses) / len(test_mses),
            "mae": sum(test_maes) / len(test_maes) if test_maes else None,
            "mape": sum(test_mapes) / len(test_mapes) if test_mapes else None,
        }

    return summary
