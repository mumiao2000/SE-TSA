# Reflection Request

Iteration: $iteration

## Best Model This Round

$best_model

## Worst Model This Round

$worst_model

## All Models This Round

$all_models

## History Summary (Last $history_rounds Rounds)

$history_summary

## MAP-Elites Coverage

Coverage: $coverage (occupied cells / total grid cells)
Occupied cells: $occupied_cells

## Failures This Round

$failures

## Existing Principles

$existing_principles

## Instructions

Analyze the results and identify the SINGLE most valuable new design principle.
Each principle should be a single concise sentence.
Return exactly 1 principle that is NOT already in the existing principles above.
If there is no genuinely new insight, return an empty list.

Return JSON: {'principles': ['principle']}
