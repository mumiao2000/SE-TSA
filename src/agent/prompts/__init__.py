"""Prompt template renderer using string.Template."""

import os
from string import Template

_PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_file(name: str) -> str:
    path = os.path.join(_PROMPTS_DIR, f"{name}.md")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_system(strategy: str = "explore") -> str:
    """Render system prompt by concatenating base + strategy-specific."""
    base = _read_file("system_base")
    strategy_file = f"system_{strategy}"
    strategy_path = os.path.join(_PROMPTS_DIR, f"{strategy_file}.md")
    if os.path.exists(strategy_path):
        strategy_text = _read_file(strategy_file)
        return f"{base}\n\n{strategy_text}"
    return base


def render(name: str, **kwargs) -> str:
    """Render a user prompt template with variable substitution."""
    text = _read_file(name)
    t = Template(text)
    return t.safe_substitute(**kwargs)
