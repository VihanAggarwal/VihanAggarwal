"""Minimal stand-ins for the omnigibson modules the vendored evaluator clients import.

Only the imports need to resolve: ``from omnigibson.macros import gm`` (network_utils reads ``gm.DEBUG`` only in
its server class, which the tests do not use) and ``from omnigibson.eval.utils.network_utils import ...``
(policies.py), which is pointed at the vendored 2026/eval network_utils module. Existing entries in sys.modules
(e.g. stubs installed by other tests) are extended, never replaced.
"""

from __future__ import annotations

import sys
import types


def _package(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        mod.__path__ = []  # behave like a package for submodule imports
        sys.modules[name] = mod
    parent_name, _, child = name.rpartition(".")
    if parent_name:
        setattr(sys.modules[parent_name], child, mod)
    return mod


def install_macros() -> None:
    _package("omnigibson")
    macros = sys.modules.get("omnigibson.macros")
    if macros is None or not hasattr(macros, "gm"):
        macros = types.ModuleType("omnigibson.macros")
        macros.gm = types.SimpleNamespace(DEBUG=False)
        sys.modules["omnigibson.macros"] = macros
        sys.modules["omnigibson"].macros = macros


def install_network_utils(module: types.ModuleType) -> None:
    """Make ``omnigibson.eval.utils.network_utils`` resolve to ``module``."""
    _package("omnigibson")
    _package("omnigibson.eval")
    _package("omnigibson.eval.utils")
    sys.modules["omnigibson.eval.utils.network_utils"] = module
    sys.modules["omnigibson.eval.utils"].network_utils = module
