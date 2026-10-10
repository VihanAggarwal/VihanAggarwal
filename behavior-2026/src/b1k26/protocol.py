"""Wire format of the BEHAVIOR evaluator's websocket policy protocol.

Byte-compatible with OmniGibson/omnigibson/eval/utils/network_utils.py (pack_data / unpack_data), which is
itself openpi's msgpack_numpy. Arrays travel as {b"__ndarray__": True, b"data", b"dtype", b"shape"} maps and
numpy scalars as {b"__npgeneric__": True, b"data", b"dtype"}. Torch tensors are converted when torch is
importable, but torch is never required.
"""

from __future__ import annotations

import functools
from typing import Any

import msgpack
import numpy as np


def pack_data(obj: Any) -> Any:
    if type(obj).__module__.startswith("torch") and hasattr(obj, "detach"):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def unpack_data(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_data)
packb = functools.partial(msgpack.packb, default=pack_data)
# The reference server unpacks with strict_map_key=False because observation dicts may carry int keys.
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_data, strict_map_key=False)
