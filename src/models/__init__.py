"""Auto-discover and expose all predefined forecasting models."""

import importlib
import os
import pkgutil
from typing import Dict, List, Type

from .base import BaseForecastModel

__all__ = ["BaseForecastModel"]

# Registry of predefined model classes
_PREDEFINED_MODELS: Dict[str, Type[BaseForecastModel]] = {}

_package_dir = os.path.dirname(__file__)


def _discover_models():
    """Auto-import all modules and register BaseForecastModel subclasses."""
    for _, module_name, ispkg in pkgutil.iter_modules([_package_dir]):
        if module_name in ("base", "registry") or ispkg:
            continue
        try:
            module = importlib.import_module(f"{__name__}.{module_name}")
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseForecastModel)
                    and attr is not BaseForecastModel
                ):
                    _PREDEFINED_MODELS[attr_name] = attr
        except Exception:
            # Skip modules that fail to import
            pass


_discover_models()

# Expose discovered classes at package level so `from src.models import X` works
for _name, _cls in _PREDEFINED_MODELS.items():
    globals()[_name] = _cls
    if _name not in __all__:
        __all__.append(_name)


def get_model_class(class_name: str) -> Type[BaseForecastModel] | None:
    """Retrieve a predefined model class by name."""
    return _PREDEFINED_MODELS.get(class_name)


def list_models() -> List[str]:
    """Return names of all predefined models."""
    return list(_PREDEFINED_MODELS.keys())
