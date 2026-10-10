"""Verbatim copies of the OmniGibson evaluator's websocket clients, loaded under private module names.

- ``load_post2()``    -> eval/utils/network_utils.py at v3.9.3-post2 (same client as v3.9.3-post1)
- ``load_2026eval()`` -> (network_utils, policies) at branch 2026/eval head 020ca52 (reconnects, deadlines,
  MultiWebsocketPolicy for ``--policy-endpoints``)
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

from . import omnigibson_stub

HERE = pathlib.Path(__file__).resolve().parent
HEADER_END = "# ---- end of b1k26 header ----\n"
VENDORED = ("og_post2_network_utils.py", "og_2026eval_network_utils.py", "og_2026eval_policies.py")


def _load(module_name: str, filename: str) -> types.ModuleType:
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, HERE / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_post2() -> types.ModuleType:
    omnigibson_stub.install_macros()
    return _load("b1k26_vendor_og_post2_network_utils", "og_post2_network_utils.py")


def load_2026eval() -> tuple[types.ModuleType, types.ModuleType]:
    omnigibson_stub.install_macros()
    nu = _load("b1k26_vendor_og_2026eval_network_utils", "og_2026eval_network_utils.py")
    omnigibson_stub.install_network_utils(nu)
    pol = _load("b1k26_vendor_og_2026eval_policies", "og_2026eval_policies.py")
    return nu, pol


def upstream_body(filename: str) -> bytes:
    """The vendored file without the b1k26 header (must be byte-identical to upstream)."""
    text = (HERE / filename).read_bytes()
    marker = HEADER_END.encode()
    idx = text.index(marker)
    return text[idx + len(marker):]


def declared_sha256(filename: str) -> str:
    for line in (HERE / filename).read_text().splitlines():
        if "sha256 " in line:
            return line.split("sha256 ", 1)[1].split(")", 1)[0].strip()
        if line == HEADER_END.strip():
            break
    raise ValueError(f"{filename}: no sha256 in header")
