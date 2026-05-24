"""Reflect and update memory after each iteration."""

import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from .client import LLMClient
from .prompts import render


def reflect(
    args: argparse.Namespace,
    stats: dict,
    llm_client: LLMClient,
    bert_model: SentenceTransformer,
) -> dict:
    """Reflect on this round's results and update principles and MAP-Elites stats."""
    iteration = stats["evolution"]["current_iteration"]
    history = stats["evolution"]["history"]
    failures = stats["evolution"]["failures"]

    # Extract results for this round
    iter_entries = [e for e in history if e.get("iteration") == iteration]
    iter_failures = [f for f in failures if f.get("iteration") == iteration]

    if not iter_entries:
        return stats

    best = min(iter_entries, key=lambda e: e["val_loss"])
    worst = max(iter_entries, key=lambda e: e["val_loss"])

    # Build prompt
    prompt = _build_reflect_prompt(
        args, stats, best, worst, iter_entries, iter_failures
    )

    try:
        response = llm_client.chat_json(
            messages=[{"role": "user", "content": prompt}],
            system="You are an expert ML researcher. Provide concise, actionable insights.",
        )
        new_principles = response.get("principles", [])
        if isinstance(new_principles, str):
            new_principles = [new_principles]

        existing_principles = stats["evolution"]["principles"]
        threshold = getattr(args, "principle_similarity_threshold", 0.85)
        added = False

        if existing_principles and new_principles:
            all_texts = existing_principles + new_principles
            embeddings = bert_model.encode(all_texts, show_progress_bar=False)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
            existing_embs = embeddings[: len(existing_principles)]

            for i, p in enumerate(new_principles):
                new_emb = embeddings[len(existing_principles) + i]
                similarities = existing_embs @ new_emb
                max_sim = float(similarities.max())
                if max_sim > threshold:
                    print(
                        f"[Reflect] Principle skipped (sim={max_sim:.3f}): {p[:60]}..."
                    )
                    continue
                stats["evolution"]["principles"].append(p)
                print(f"[Reflect] Added new principle (sim={max_sim:.3f}): {p[:60]}...")
                added = True
                break

            if not added:
                print("[Reflect] All new principles are too similar to existing ones.")
        else:
            for p in new_principles:
                stats["evolution"]["principles"].append(p)
                print(f"[Reflect] Added new principle: {p[:60]}...")
                added = True
                break

        stats["evolution"]["principles"] = stats["evolution"]["principles"][
            -args.max_principles :
        ]
        print(f"[Reflect] Total principles: {len(stats['evolution']['principles'])}.")
    except Exception as exc:
        print(f"[Reflect] step failed: {exc}")

    # Coverage update is handled in _update_map_elites_cell
    # exploration_rate decay is handled in run.py
    return stats


def _build_history_summary(
    stats: dict, current_iteration: int, n_rounds: int = 5
) -> str:
    """Build a history summary of the last N rounds for the reflection prompt."""
    history = stats["evolution"]["history"]
    map_elites = stats["evolution"]["map_elites"]
    cells = map_elites.get("cells", {})

    lines = []
    lines.append("| Round | Best Arch | Avg Val Loss | New Cells | Coverage |")
    lines.append("|-------|-----------|--------------|-----------|----------|")

    has_data = False
    for r in range(max(0, current_iteration - n_rounds + 1), current_iteration + 1):
        round_entries = [e for e in history if e.get("iteration") == r]
        if not round_entries:
            continue
        has_data = True

        # Best arch family this round
        best_entry = min(round_entries, key=lambda e: e.get("val_loss", float("inf")))
        best_arch = best_entry.get("arch_family", "N/A")

        # Average val loss
        losses = [
            e.get("val_loss", 0)
            for e in round_entries
            if isinstance(e.get("val_loss"), (int, float))
        ]
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        # New occupied cells this round (cells whose best_entry iteration == r)
        new_cells = 0
        for key, cell in cells.items():
            if (
                cell.get("occupied")
                and cell.get("best_entry", {}).get("iteration") == r
            ):
                new_cells += 1

        # Coverage (we don't have per-round coverage stored, use current for the latest)
        coverage_str = "N/A"
        if r == current_iteration:
            coverage = stats["evolution"]["map_elites"].get("coverage", 0.0)
            coverage_str = f"{coverage:.2%}"

        lines.append(
            f"| {r} | {best_arch} | {avg_loss:.6f} | {new_cells} | {coverage_str} |"
        )

    if not has_data:
        return "No historical data available yet."

    return "\n".join(lines)


def _build_reflect_prompt(args, stats, best, worst, iter_entries, iter_failures) -> str:
    """Build the reflection prompt from template."""
    map_elites = stats["evolution"]["map_elites"]
    principles = stats["evolution"]["principles"]
    current_iteration = stats["evolution"]["current_iteration"]

    all_models = [
        {
            "class_name": e["class_name"],
            "val_loss": e["val_loss"],
            "num_params": e["num_params"],
            "arch_family": e["arch_family"],
        }
        for e in iter_entries
    ]

    occupied = sum(1 for c in map_elites["cells"].values() if c["occupied"])
    history_summary = _build_history_summary(stats, current_iteration, n_rounds=5)

    return render(
        "reflect",
        iteration=current_iteration,
        best_model=json.dumps(best, indent=2, default=str),
        worst_model=json.dumps(worst, indent=2, default=str),
        all_models=json.dumps(all_models, indent=2),
        history_rounds=5,
        history_summary=history_summary,
        coverage=f"{map_elites['coverage']:.2%}",
        occupied_cells=occupied,
        failures=json.dumps(iter_failures, indent=2, default=str)
        if iter_failures
        else "None",
        existing_principles="\n".join(f"- {p}" for p in principles)
        if principles
        else "None yet.",
    )
