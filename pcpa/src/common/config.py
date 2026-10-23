# ---------------------------------------------------------------------
# Single source of truth for every experiment is config/default.yaml.
# A run loads it, overrides a few keys in memory, and writes the fully
# resolved configuration next to its outputs, so any number in the paper
# can be traced back to the parameters that produced it.
# ---------------------------------------------------------------------

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """The project root, i.e. the directory that contains config/default.yaml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "default.yaml").is_file():
            return parent
    raise FileNotFoundError(f"config/default.yaml not found above {here}")


class Config(dict):
    """A dict that also supports dotted lookups.

    ``cfg["contact.pacejka.B"]`` and ``cfg.get_path("contact.pacejka.B")``
    are equivalent; ``cfg["contact"]["pacejka"]["B"]`` keeps working too.
    """

    def __getitem__(self, key: str) -> Any:
        if isinstance(key, str) and "." in key and key not in self.keys():
            return self.get_path(key)
        return super().__getitem__(key)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is None:
                    raise KeyError(f"no such configuration key: {dotted}")
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> "Config":
        parts = dotted.split(".")
        node: Any = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        return self

    def resolve_paths(self, root: Path | None = None) -> "Config":
        """Make every entry of the ``paths`` block absolute."""
        root = Path(root) if root else project_root()
        for key, value in list(self.setdefault("paths", {}).items()):
            if isinstance(value, (str, os.PathLike)):
                candidate = Path(value)
                if not candidate.is_absolute():
                    self["paths"][key] = root / candidate
        return self

    def frozen(self) -> dict:
        """A plain, deep copy with PosixPath/WindowsPath turned into str."""
        def plain(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: plain(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [plain(v) for v in obj]
            if isinstance(obj, Path):
                return str(obj)
            return obj

        return plain(copy.deepcopy(dict(self)))


def load_config(path: str | os.PathLike | None = None, *, resolve: bool = True) -> Config:
    """Load the YAML configuration.

    ``path`` may point at a file, at a directory containing ``default.yaml``,
    or be omitted, in which case ``config/default.yaml`` is used.
    """
    root = project_root()
    if path is None:
        target = root / "config" / "default.yaml"
    else:
        target = Path(path)
        if target.is_dir():
            target = target / "default.yaml"
    if not target.is_file():
        raise FileNotFoundError(f"configuration not found: {target}")
    with open(target, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    cfg = Config(data)
    if resolve:
        cfg.resolve_paths(root)
    return cfg
