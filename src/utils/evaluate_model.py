"""Model training and evaluation (pure functions, no hard-coded defaults)."""

import importlib.util
import itertools
import multiprocessing as mp
import os
import random
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _make_failure_result(status: str, error_message: str) -> dict:
    return {
        "status": status,
        "mse": float("inf"),
        "mae": float("inf"),
        "mape": float("inf"),
        "training_time": 0.0,
        "num_params": 0,
        "epochs_trained": 0,
        "best_config": None,
        "error_message": error_message,
    }


def _evaluate_worker(
    model_path: str,
    class_name: str,
    config: dict | None,
    train_xy: tuple[np.ndarray, np.ndarray],
    eval_xy: tuple[np.ndarray, np.ndarray],
    eval_cfg: dict,
    return_predictions: bool,
    queue: "mp.Queue",
) -> None:
    """Run inside a spawned subprocess; re-imports the model from file."""
    try:
        proj_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if proj_root not in sys.path:
            sys.path.insert(0, proj_root)

        spec = importlib.util.spec_from_file_location(
            f"_subproc_{class_name}", model_path
        )
        if spec is None or spec.loader is None:
            queue.put(
                _make_failure_result(
                    "error",
                    f"Could not load module spec from {model_path}",
                )
            )
            return
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        model_cls = getattr(mod, class_name)

        result = _evaluate_impl(
            model_cls, config, train_xy, eval_xy, eval_cfg, return_predictions
        )
        queue.put(result)
    except BaseException as exc:  # noqa: BLE001
        queue.put(_make_failure_result("error", f"{type(exc).__name__}: {exc}"))


def evaluate_model(
    model_path: str,
    class_name: str,
    config: dict | None,
    train_xy: tuple[np.ndarray, np.ndarray],
    eval_xy: tuple[np.ndarray, np.ndarray],
    eval_cfg: dict,
    return_predictions: bool = False,
) -> dict:
    """Run training and evaluation in a fresh spawn subprocess.

    Failure statuses: ``cuda_fatal`` (subprocess killed, likely CUDA assert),
    ``timeout`` (forced kill after time budget), ``error`` (normal exception
    inside worker).
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_evaluate_worker,
        args=(
            model_path,
            class_name,
            config,
            train_xy,
            eval_xy,
            eval_cfg,
            return_predictions,
            queue,
        ),
    )
    proc.start()

    # quick_search_max_time is in minutes; allow ~2x budget plus a buffer for
    # spawn startup, model import, and dataloader workers.
    budget_minutes = float(eval_cfg.get("quick_search_max_time", 3.0))
    timeout_seconds = budget_minutes * 60.0 * 2.0 + 60.0
    proc.join(timeout=timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return _make_failure_result(
            "timeout",
            f"Subprocess timeout after {timeout_seconds:.0f}s",
        )

    if proc.exitcode != 0:
        try:
            return queue.get(timeout=1)
        except Exception:
            return _make_failure_result(
                "cuda_fatal",
                f"Subprocess died (exitcode={proc.exitcode}); "
                f"likely CUDA device-side assert triggered inside the model.",
            )

    try:
        return queue.get(timeout=5)
    except Exception as exc:
        return _make_failure_result(
            "error",
            f"Subprocess exited cleanly but produced no result: {exc}",
        )


def _evaluate_impl(
    model_cls: type,
    config: dict | None,
    train_xy: tuple[np.ndarray, np.ndarray],
    eval_xy: tuple[np.ndarray, np.ndarray],
    eval_cfg: dict,
    return_predictions: bool = False,
) -> dict:
    """
    Train and evaluate the model. Runs in the worker subprocess.

    1. If config is None and model_cls has get_all_configs() -> serial quick search for the best config
    2. Train on train_xy with the best config (or given config), early-stop on eval_xy
       (also stops once quick_search_max_time elapses)
    3. Compute mse, mae, mape on eval_xy (column 0 only)
    """
    result = {
        "status": "error",
        "mse": float("inf"),
        "mae": float("inf"),
        "mape": float("inf"),
        "training_time": 0.0,
        "num_params": 0,
        "epochs_trained": 0,
        "best_config": None,
        "error_message": None,
    }

    train_x, train_y = train_xy
    eval_x, eval_y = eval_xy
    input_len = eval_cfg["input_len"]
    output_len = eval_cfg["output_len"]
    n_features = train_x.shape[2]

    device_str = eval_cfg.get("device", "auto")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and device_str == "auto"
        else (device_str if device_str != "auto" else "cpu")
    )

    batch_size = eval_cfg["batch_size"]
    lr = eval_cfg["lr"]
    epochs = eval_cfg["train_max_epochs"]
    patience = eval_cfg["train_patience"]

    train_dataset = TensorDataset(
        torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()
    )
    eval_dataset = TensorDataset(
        torch.from_numpy(eval_x).float(), torch.from_numpy(eval_y).float()
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    try:
        best_config = config
        if (
            config is None
            and hasattr(model_cls, "get_all_configs")
            and callable(model_cls.get_all_configs)
        ):
            search_space = model_cls.get_all_configs()
            if search_space:
                best_config = _search_best_config(
                    model_cls,
                    train_loader=train_loader,
                    eval_loader=eval_loader,
                    device=device,
                    input_len=input_len,
                    output_len=output_len,
                    n_features=n_features,
                    eval_cfg=eval_cfg,
                )

        if best_config is not None:
            model = model_cls(input_len, output_len, n_features, best_config).to(device)
        else:
            model = model_cls(input_len, output_len, n_features).to(device)

        num_params = model.count_parameters()

        final_budget = eval_cfg.get("quick_search_max_time", 3.0) * 60
        start_time = time.time()
        _, epochs_trained = train_one_model(
            model,
            train_loader,
            eval_loader,
            device,
            epochs,
            lr,
            patience,
            time_budget_seconds=final_budget,
        )
        training_time = time.time() - start_time

        if return_predictions:
            mse, mae, mape, pred_arr, target_arr = eval_metrics(
                model,
                eval_loader,
                device,
                eval_cfg["mape_threshold"],
                return_predictions=True,
            )
            result["predictions"] = pred_arr
            result["targets"] = target_arr
        else:
            mse, mae, mape = eval_metrics(
                model, eval_loader, device, eval_cfg["mape_threshold"]
            )

        result.update(
            {
                "status": "success",
                "mse": mse,
                "mae": mae,
                "mape": mape,
                "training_time": training_time,
                "num_params": num_params,
                "epochs_trained": epochs_trained,
                "best_config": best_config,
                "error_message": None,
            }
        )

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        result["status"] = "oom"
        result["error_message"] = "CUDA out of memory"
    except Exception as exc:
        result["status"] = "error"
        result["error_message"] = str(exc)
    finally:
        try:
            del train_loader
            del eval_loader
        except NameError:
            pass

    return result


def _search_best_config(
    model_cls,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    input_len: int,
    output_len: int,
    n_features: int,
    eval_cfg: dict,
) -> Optional[Dict[str, Any]]:
    all_configs = model_cls.get_all_configs()
    if not all_configs:
        return None

    keys = list(all_configs.keys())
    values = list(all_configs.values())
    combinations = list(itertools.product(*values))

    max_combos = eval_cfg.get("quick_search_max_combinations", 8)
    if len(combinations) > max_combos:
        random.shuffle(combinations)
        combinations = combinations[:max_combos]

    lr = eval_cfg["lr"]
    epochs = eval_cfg["quick_search_max_epochs"]
    patience = eval_cfg["quick_search_patience"]

    best_mse = float("inf")
    best_config = None
    for combo in combinations:
        config = dict(zip(keys, combo))
        try:
            model = model_cls(input_len, output_len, n_features, config).to(device)
            train_one_model(
                model, train_loader, eval_loader, device, epochs, lr, patience
            )
            mse, _, _ = eval_metrics(
                model, eval_loader, device, eval_cfg["mape_threshold"]
            )
            if mse < best_mse:
                best_mse = mse
                best_config = config
        except Exception:
            continue
    return best_config


def train_one_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    patience,
    time_budget_seconds: float | None = None,
) -> tuple[float, int]:
    """Standard training loop; returns best_val_loss and actual epochs trained.

    When ``time_budget_seconds`` is set, training also stops once cumulative
    elapsed time exceeds the budget (checked after each epoch). The returned
    ``best_val_loss`` is always the best observed, even on time-budget exit.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    best_val_loss = float("inf")
    epochs_trained = 0
    patience_counter = 0
    best_state = None
    start = time.time()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())
        avg_val_loss = float(np.mean(val_losses))
        epochs_trained = epoch + 1

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if (
            time_budget_seconds is not None
            and (time.time() - start) > time_budget_seconds
        ):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_loss, epochs_trained


def eval_metrics(
    model, eval_loader, device, mape_threshold: float, return_predictions: bool = False
):
    """Compute mse, mae, mape on eval_loader (column 0 only)."""
    mse_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in eval_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            all_preds.append(pred)
            all_targets.append(yb)

    pred = torch.cat(all_preds, dim=0)
    target = torch.cat(all_targets, dim=0)

    pred_c0 = pred[:, :, 0]
    target_c0 = target[:, :, 0]

    mse = mse_fn(pred_c0, target_c0).item()
    mae = mae_fn(pred_c0, target_c0).item()

    mask = torch.abs(target_c0) > mape_threshold
    if mask.sum() > 0:
        mape = (
            torch.mean(
                torch.abs((pred_c0[mask] - target_c0[mask]) / target_c0[mask])
            ).item()
            * 100
        )
    else:
        mape = float("nan")

    if return_predictions:
        return mse, mae, mape, pred.cpu().numpy(), target.cpu().numpy()
    return mse, mae, mape
