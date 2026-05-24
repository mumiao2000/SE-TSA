# Model Generation Request (EXPLORE)

Iteration: $iteration

## Dataset Statistics

$data_stats

## Empty MAP-Elites Cells (prioritize these)

$empty_cells

## Validated Design Principles

$principles

## Recent Failures to Avoid

$failures

## Instructions

Generate $n_models diverse model designs for this dataset.
Each design should explore different trade-offs (accuracy vs params vs speed).
In addition to improving current architectures, actively try NEW architecture families or novel combinations that have not been explored yet.
Be active to design channel dependent models if the data has multiple features.

1. First, briefly describe your design approach in 2-3 sentences inside the `design_notes` field.
2. Then, write the complete model code.

Return JSON with a single key 'models', which is a list of objects:
{
  "models": [
    {
      "class_name": "ModelName",
      "description": "One-line summary",
      "design_rationale": "Why this should work",
      "design_notes": "Brief design approach in 2-3 sentences",
      "arch_family": "linear|attention|hybrid|advanced",
      "preprocessing": "revin|hybrid|advanced",
      "code": "class ModelName(BaseForecastModel): ..."
    }
  ]
}
