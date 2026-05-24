# Error Correction Request

Model: $class_name

## Original Description

$description

## Original Design Rationale

$design_rationale

## Original Arch Family

$arch_family

## Prior Failed Fix Attempts

$prior_attempts

## Evaluation Error (latest)

```
$error_message
```

## Broken Model Code (latest)

```python
$code
```

## Task

Fix the error in the model code. The fixed model MUST:

1. Keep the EXACT same class name
2. Inherit from `BaseForecastModel`
3. Accept `(self, input_len, output_len, n_features, config=None)` in `__init__`
4. Return shape `(batch_size, output_len, n_features)` from `forward()`
5. Use `self.input_len`, `self.output_len`, and `self.n_features`
6. Only use `torch` and standard Python libraries
7. Preserve the original architecture family and design intent

Return JSON with a single key 'models' containing one fixed model object:
{
  "models": [
    {
      "class_name": "$class_name",
      "description": "Fixed model",
      "design_rationale": "Why the fix works",
      "design_notes": "Brief explanation of the fix",
      "arch_family": "linear|attention|hybrid|advanced",
      "preprocessing": "revin|hybrid|advanced",
      "code": "class $class_name(BaseForecastModel): ..."
    }
  ]
}
