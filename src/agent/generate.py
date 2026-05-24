"""Generate model library: explore / exploit, with validation."""

import argparse
import importlib.util
import json
import os
import random
import sys

import torch

from ..utils.read_model import list_predefined_models, read_predefined_model_source
from .client import LLMClient
from .prompts import render, render_system


def generate_model_library(
    args: argparse.Namespace,
    stats: dict,
    llm_client: LLMClient,
) -> list[dict]:
    """Generate model library: return predefined models at iteration 0, then explore/exploit."""
    iteration = stats["evolution"]["current_iteration"]

    if iteration == 0:
        return _load_predefined_models(args, stats)

    strategy = _decide_strategy(stats)
    print(f"[Generator] Strategy: {strategy}")

    if strategy == "explore":
        raw_models = _explore(args, stats, llm_client)
    else:
        raw_models = _exploit(args, stats, llm_client)

    candidates = []
    for model_dict in raw_models:
        validated = _validate_and_save_model(model_dict, args, stats)
        if validated is not None:
            candidates.append(validated)
        else:
            # Try to fix
            fixed = _try_fix_model(model_dict, args, stats, llm_client)
            if fixed is not None:
                candidates.append(fixed)
            else:
                stats["evolution"]["failures"].append(
                    {
                        "iteration": iteration,
                        "class_name": model_dict.get(
                            "_final_failed_class_name",
                            model_dict.get("class_name", "Unknown"),
                        ),
                        "status": "validation_failed",
                        "error_message": model_dict.get(
                            "_final_error_traceback",
                            model_dict.get(
                                "_error_traceback",
                                "Model failed validation after retries",
                            ),
                        ),
                        "timestamp": __import__("time").time(),
                    }
                )

    return candidates


def _load_predefined_models(args, stats) -> list[dict]:
    """Load predefined models, validate, and save them to the models/ directory."""
    iteration = stats["evolution"]["current_iteration"]
    model_names = list_predefined_models()
    candidates = []

    models_dir = os.path.join(args.output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    input_len = args.input_len
    output_len = args.output_len
    n_features = stats["data"]["n_features"]

    for name in model_names:
        source = read_predefined_model_source(name)
        if source is None:
            continue

        # Save to models/{name}_0.py
        code_file = os.path.join(models_dir, f"{name}_{iteration}.py")
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(_build_model_file_content(source))

        # Dynamic import validation
        try:
            spec = importlib.util.spec_from_file_location(
                f"_predefined_{name}_{iteration}", code_file
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"_predefined_{name}_{iteration}"] = mod
            spec.loader.exec_module(mod)
            ModelCls = getattr(mod, name)

            # Note: dynamic import may create a different BaseForecastModel class object,
            # so we skip strict issubclass check and rely on dummy-input validation.

            # Dummy input validation
            device = torch.device("cpu")
            model = ModelCls(input_len, output_len, n_features).to(device)
            dummy = torch.randn(2, input_len, n_features).to(device)
            out = model(dummy)
            if out.shape != (2, output_len, n_features):
                raise ValueError(
                    f"Output shape {out.shape} != {(2, output_len, n_features)}"
                )

            arch_family = _infer_arch_family(source)
            preprocessing = _infer_preprocessing(source)

            candidates.append(
                {
                    "class_name": name,
                    "model_cls": ModelCls,
                    "model_path": code_file,
                    "description": f"Predefined {name}",
                    "design_rationale": "Built-in baseline model",
                    "arch_family": arch_family,
                    "preprocessing": preprocessing,
                }
            )
            print(f"[Generator] Predefined {name} validated and saved.")
        except Exception as exc:
            print(f"[Generator] Predefined {name} validation failed: {exc}")
            stats["evolution"]["failures"].append(
                {
                    "iteration": iteration,
                    "class_name": name,
                    "status": "validation_failed",
                    "error_message": str(exc),
                    "timestamp": __import__("time").time(),
                }
            )

    return candidates


def _decide_strategy(stats) -> str:
    """Decide between EXPLORE and EXPLOIT."""
    exploration_rate = stats["evolution"]["exploration_rate"]
    coverage = stats["evolution"]["map_elites"]["coverage"]

    if coverage < 1.0:
        # Higher empty-cell ratio increases the probability of explore
        empty_ratio = 1.0 - coverage
        explore_prob = max(
            exploration_rate, empty_ratio / 5
        )  # Cap at 20% explore if there are many empty cells
        return "explore" if random.random() < explore_prob else "exploit"

    return "explore" if random.random() < exploration_rate else "exploit"


def _explore(args, stats, llm_client) -> list[dict]:
    """Call LLM to generate brand-new architecture models."""
    n_candidates = args.n_candidates

    prompt = _build_explore_prompt(args, stats, n_candidates)
    try:
        response = llm_client.chat_json(
            messages=[{"role": "user", "content": prompt}],
            system=render_system("explore"),
        )
        models = response.get("models", [])
        if not isinstance(models, list):
            models = [models]
        return [_sanitize_model(m) for m in models]
    except Exception as exc:
        print(f"[Generator] Explore LLM call failed: {exc}")
        return []


def _exploit(args, stats, llm_client) -> list[dict]:
    """Improve upon the best models in the MAP-Elites grid."""
    grid = stats["evolution"]["map_elites"]
    cells = grid["cells"]

    # Group by arch_family and pick the best from each group
    family_best = {}
    for key, cell in cells.items():
        if not cell["occupied"] or cell["best_entry"] is None:
            continue
        af = cell["best_entry"]["arch_family"]
        if af not in family_best or cell["best_val_loss"] < family_best[af]["val_loss"]:
            family_best[af] = cell["best_entry"]

    if not family_best:
        return []

    # Take top-k distinct families
    sorted_entries = sorted(family_best.values(), key=lambda e: e["val_loss"])
    parents = sorted_entries[: args.exploit_top_k]

    all_models = []
    for parent in parents:
        prompt = _build_exploit_prompt(args, stats, parent)
        try:
            response = llm_client.chat_json(
                messages=[{"role": "user", "content": prompt}],
                system=render_system("exploit"),
            )
            models = response.get("models", [])
            if not isinstance(models, list):
                models = [models]
            all_models.extend([_sanitize_model(m) for m in models])
        except Exception as exc:
            print(
                f"[Generator] Exploit LLM call failed for {parent['class_name']}: {exc}"
            )

    return all_models


def _validate_and_save_model(model_dict, args, stats) -> dict | None:
    """Validate the model and save it to the models/ directory."""
    iteration = stats["evolution"]["current_iteration"]
    class_name = model_dict.get("class_name", "GeneratedModel")
    code = model_dict.get("code", "")

    if not code:
        return None

    models_dir = os.path.join(args.output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    code_file = os.path.join(models_dir, f"{class_name}_{iteration}.py")

    with open(code_file, "w", encoding="utf-8") as f:
        f.write(_build_model_file_content(code))

    input_len = args.input_len
    output_len = args.output_len
    n_features = stats["data"]["n_features"]

    try:
        spec = importlib.util.spec_from_file_location(
            f"_gen_{class_name}_{iteration}", code_file
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"_gen_{class_name}_{iteration}"] = mod
        spec.loader.exec_module(mod)
        ModelCls = getattr(mod, class_name)

        device = torch.device("cpu")
        model = ModelCls(input_len, output_len, n_features).to(device)
        dummy = torch.randn(2, input_len, n_features).to(device)
        out = model(dummy)
        if out.shape != (2, output_len, n_features):
            raise ValueError(
                f"Output shape {out.shape} != {(2, output_len, n_features)}"
            )

        return {
            "class_name": class_name,
            "model_cls": ModelCls,
            "model_path": code_file,
            "description": model_dict.get("description", ""),
            "design_rationale": model_dict.get("design_rationale", ""),
            "arch_family": model_dict.get("arch_family", "hybrid"),
            "preprocessing": model_dict.get("preprocessing", "none"),
        }
    except Exception as exc:
        import traceback

        tb = traceback.format_exc()
        print(f"[Generator] Validation failed for {class_name}: {exc}")
        model_dict["_error_traceback"] = f"{exc}\n\n{tb}"
        try:
            os.remove(code_file)
        except OSError:
            pass
        return None


def _try_fix_model(model_dict, args, stats, llm_client) -> dict | None:
    """Try to fix a failed model using the LLM, accumulating attempt history."""
    retries = getattr(args, "n_fix_retries", 1)
    original_class_name = model_dict.get("class_name", "GeneratedModel")

    current_class_name = original_class_name
    current_code = model_dict.get("code", "")
    current_error = model_dict.get(
        "_error_traceback", "Validation failed: shape mismatch or runtime error"
    )
    prior_attempts: list[dict] = []

    for attempt in range(retries):
        prompt = _build_fix_prompt(
            current_class_name,
            current_code,
            current_error,
            prior_attempts=prior_attempts,
            description=model_dict.get("description", ""),
            design_rationale=model_dict.get("design_rationale", ""),
            arch_family=model_dict.get("arch_family", "hybrid"),
        )
        try:
            response = llm_client.chat_json(
                messages=[{"role": "user", "content": prompt}],
                system=render_system("fix"),
            )
            models = response.get("models", [])
            if not models:
                continue
            fixed = _sanitize_model(models[0])
            new_name = f"{original_class_name}_Fixed{attempt}"
            fixed["class_name"] = new_name
            old_decl = f"class {current_class_name}("
            new_decl = f"class {new_name}("
            fixed["code"] = fixed.get("code", "").replace(old_decl, new_decl, 1)
            validated = _validate_and_save_model(fixed, args, stats)
            if validated is not None:
                return validated

            prior_attempts.append(
                {
                    "class_name": current_class_name,
                    "code": current_code,
                    "error": current_error,
                }
            )
            current_class_name = new_name
            current_code = fixed.get("code", current_code)
            current_error = fixed.get("_error_traceback", current_error)
        except Exception as exc:
            print(f"[Generator] Fix attempt {attempt} failed: {exc}")

    model_dict["_final_error_traceback"] = current_error
    model_dict["_final_failed_class_name"] = current_class_name
    return None


def _build_model_file_content(code: str) -> str:
    """Add necessary imports to the model code."""
    lines = [
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "from src.models.base import BaseForecastModel",
        "from src.models.layers.revin import RevIN",
        "",
        code,
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builders (render from external templates)
# ---------------------------------------------------------------------------


def _translate_empty_cells(empty_cells: list[str]) -> str:
    """Translate raw MAP-Elites cell keys to human-readable descriptions."""
    if not empty_cells:
        return "None"

    descriptions = []
    for cell in empty_cells[:20]:
        parts = cell.split("|")
        if len(parts) >= 2:
            arch, preproc = parts[0], parts[1]
            descriptions.append(f"- {arch} + {preproc}")
        else:
            descriptions.append(f"- {cell}")

    return "\n".join(descriptions) if descriptions else "None"


def _get_same_family_models(stats: dict, parent: dict, top_k: int = 3) -> str:
    """Get top models in the same architecture family as the parent."""
    cells = stats["evolution"]["map_elites"]["cells"]
    parent_af = parent.get("arch_family", "hybrid")
    entries = []
    for key, cell in cells.items():
        if not cell.get("occupied") or cell.get("best_entry") is None:
            continue
        entry = cell["best_entry"]
        if entry.get("arch_family") == parent_af:
            entries.append(entry)

    entries.sort(key=lambda e: e.get("val_loss", float("inf")))
    if not entries:
        return "None"

    lines = []
    for e in entries[:top_k]:
        val_loss = e.get("val_loss")
        num_params = e.get("num_params", "N/A")
        vl_str = (
            f"{val_loss:.6f}" if isinstance(val_loss, (int, float)) else str(val_loss)
        )
        lines.append(f"- {e['class_name']}: val_loss={vl_str}, params={num_params}")
        dr = e.get("design_rationale", "")
        if dr:
            lines.append(
                f"  Design rationale: {dr[:120]}{'...' if len(dr) > 120 else ''}"
            )
    return "\n".join(lines)


def _get_global_best_model(stats: dict) -> str:
    """Get the global best model across all history."""
    history = stats["evolution"]["history"]
    if not history:
        return "None"
    best = min(history, key=lambda e: e.get("val_loss", float("inf")))
    return json.dumps(best, indent=2, default=str)


def _get_failed_variants(stats: dict, parent_class_name: str, limit: int = 3) -> str:
    """Get recent failed variants related to the parent model."""
    failures = stats["evolution"]["failures"]
    related = []
    for f in failures:
        cls_name = f.get("class_name", "")
        if parent_class_name in cls_name or cls_name in parent_class_name:
            related.append(f)
    if not related:
        return "None"

    lines = []
    for f in related[-limit:]:
        msg = f.get("error_message", "Unknown error")[:100]
        lines.append(
            f"- {f['class_name']}: {msg}{'...' if len(f.get('error_message', '')) > 100 else ''}"
        )
    return "\n".join(lines)


def _build_explore_prompt(args, stats, n_models: int) -> str:
    """Build the explore prompt from template."""
    data = stats.get("data", {})
    map_elites = stats["evolution"]["map_elites"]
    principles = stats["evolution"]["principles"]
    failures = stats["evolution"]["failures"]
    empty_cells = [k for k, v in map_elites["cells"].items() if not v["occupied"]]
    empty_cells_text = _translate_empty_cells(empty_cells)

    principles_text = (
        "\n".join(f"- {p}" for p in principles) if principles else "None yet."
    )
    failures_text = (
        "\n".join(f"- {f['class_name']}: {f['error_message']}" for f in failures[-5:])
        if failures
        else "None yet."
    )

    return render(
        "explore",
        iteration=stats["evolution"]["current_iteration"],
        data_stats=json.dumps(data, indent=2),
        empty_cells=empty_cells_text,
        principles=principles_text,
        failures=failures_text,
        n_models=n_models,
    )


def _build_exploit_prompt(args, stats, parent: dict) -> str:
    """Build the exploit prompt from template."""
    data = stats.get("data", {})
    principles = stats["evolution"]["principles"]

    val_loss = parent.get("val_loss", float("inf"))
    val_loss_str = (
        f"{val_loss:.6f}" if isinstance(val_loss, (int, float)) else str(val_loss)
    )

    same_family = _get_same_family_models(stats, parent, top_k=3)
    global_best = _get_global_best_model(stats)
    failed_variants = _get_failed_variants(stats, parent["class_name"], limit=3)

    principles_text = (
        "\n".join(f"- {p}" for p in principles) if principles else "None yet."
    )

    return render(
        "exploit",
        iteration=stats["evolution"]["current_iteration"],
        data_stats=json.dumps(data, indent=2),
        parent_class_name=parent["class_name"],
        parent_val_loss=val_loss_str,
        parent_num_params=parent.get("num_params", "N/A"),
        parent_arch_family=parent.get("arch_family", "hybrid"),
        parent_description=parent.get("description", ""),
        parent_design_rationale=parent.get("design_rationale", ""),
        same_family_models=same_family,
        global_best_model=global_best,
        failed_variants=failed_variants,
        principles=principles_text,
    )


def _build_fix_prompt(
    class_name: str,
    code: str,
    error_message: str,
    prior_attempts: list[dict] | None = None,
    description: str = "",
    design_rationale: str = "",
    arch_family: str = "hybrid",
) -> str:
    """Build the fix prompt from template."""
    return render(
        "fix",
        class_name=class_name,
        description=description or "No description available.",
        design_rationale=design_rationale or "No design rationale available.",
        arch_family=arch_family,
        error_message=error_message,
        code=code,
        prior_attempts=_format_prior_attempts(prior_attempts or []),
    )


def _format_prior_attempts(attempts: list[dict]) -> str:
    """Render prior fix attempts as a markdown block for the fix prompt."""
    if not attempts:
        return "None (this is the first fix attempt)."
    blocks = []
    for i, a in enumerate(attempts):
        blocks.append(
            f"### Attempt {i}: `{a['class_name']}`\n\n"
            f"Error:\n```\n{a['error']}\n```\n\n"
            f"Code:\n```python\n{a['code']}\n```"
        )
    return "\n\n".join(blocks)


def _sanitize_model(model: dict) -> dict:
    """Clean and normalize the model dictionary."""
    code = model.get("code", "")
    arch = model.get("arch_family", "")
    preproc = model.get("preprocessing", "")
    if not arch:
        arch = _infer_arch_family(code)
    if not preproc:
        preproc = _infer_preprocessing(code)
    model["arch_family"] = arch.lower() if arch else "hybrid"
    model["preprocessing"] = preproc.lower() if preproc else "hybrid"
    return model


def _infer_arch_family(code: str) -> str:
    code_lower = code.lower()
    advanced_markers = (
        "mamba",
        "selective_scan",
        "selectivescan",
        "s4layer",
        "s5layer",
        "kanlayer",
        "kan_layer",
        "kolmogorov",
        "neural_ode",
        "neuralode",
        "torchdiffeq",
        "odeint",
        "fno",
        "fourierneuraloperator",
        "spectralconv",
        "liquid",
        "ltcell",
    )
    if any(tok in code_lower for tok in advanced_markers):
        return "advanced"

    has_lstm = "lstm" in code_lower
    has_gru = "gru" in code_lower
    has_conv = "conv1d" in code_lower or "conv2d" in code_lower
    has_attn = (
        "attention" in code_lower
        or "transformer" in code_lower
        or "multihead" in code_lower
    )
    has_linear = "linear" in code_lower

    families = []
    if has_lstm or has_gru:
        families.append("rnn")
    if has_conv:
        families.append("cnn")
    if has_attn:
        families.append("attention")
    if has_linear:
        families.append("linear")

    return "hybrid" if len(families) > 1 else (families[0] if families else "linear")


def _infer_preprocessing(code: str) -> str:
    code_lower = code.lower()
    matched = []
    if "revin" in code_lower:
        matched.append("revin")
    if (
        "batchnorm" in code_lower
        or "layernorm" in code_lower
        or "instancenorm" in code_lower
        or "normalize" in code_lower
    ):
        matched.append("instancenorm")
    if (
        "fft" in code_lower
        or "wavelet" in code_lower
        or "kernel_pca" in code_lower
        or "fourier_transform" in code_lower
    ):
        matched.append("advanced")
    if len(matched) >= 2:
        return "hybrid"
    if len(matched) == 1:
        return matched[0]
    return "instancenorm"
