"""Read predefined model sources from src/models/."""

import os


def read_predefined_model_source(class_name: str) -> str | None:
    """Read model source code string from src/models/{class_name}.py."""
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "models", f"{class_name}.py"
    )
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        return None
    with open(model_path, "r", encoding="utf-8") as f:
        return f.read()


def list_predefined_models() -> list[str]:
    """Scan the src/models/ directory and return a list of model class names excluding base/layers."""
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    models_dir = os.path.abspath(models_dir)
    names = []
    for fname in os.listdir(models_dir):
        if not fname.endswith(".py"):
            continue
        if fname in ("base.py", "__init__.py"):
            continue
        name = fname[:-3]
        names.append(name)
    return sorted(names)
