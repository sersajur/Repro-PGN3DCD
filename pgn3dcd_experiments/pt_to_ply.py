"""Convert a preprocessed HKCD cloud (pc0_<i>.pt / pc1_<i>.pt) to PLY for CloudCompare.

The .pt is a torch_geometric Data saved with torch.save, holding the
grid-subsampled cloud the model actually trains on:
    pos [N,3] float, f [N,3] RGB in [0,1], y [N] cd_type label.

Writes a binary PLY with x,y,z, red/green/blue (uint8) and a scalar field
"scalar_cd_type" (the label) — CloudCompare shows the RGB colours and lets you
colour-by the cd_type scalar field.

Usage:
    python pt_to_ply.py <input.pt> <output.ply>
"""
import sys

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def _load_data(path):
    """Load the .pt whether or not torch_geometric is importable.

    Uses the real package when present; otherwise registers a minimal shim so
    torch.load can unpickle the Data/storage objects (e.g. in the .venv that has
    torch + plyfile but not torch_geometric).
    """
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        import types

        def mk(name):
            m = types.ModuleType(name)
            sys.modules[name] = m
            return m

        class _Capture:
            def __init__(self, *a, **k):
                self.__dict__.update(k)

            def __setstate__(self, s):
                self.__dict__.update(s if isinstance(s, dict) else getattr(s, "__dict__", {}))

        tg = mk("torch_geometric")
        tg_data = mk("torch_geometric.data")
        tg_dd = mk("torch_geometric.data.data")
        tg_storage = mk("torch_geometric.data.storage")
        tg.data = tg_data
        for mod in (tg_data, tg_dd):
            mod.Data = type("Data", (_Capture,), {})
        tg_storage.GlobalStorage = type("GlobalStorage", (_Capture,), {})
        tg_storage.BaseStorage = type("BaseStorage", (_Capture,), {})

    return torch.load(path, map_location="cpu", weights_only=False)


def _tensors(data):
    out = {}

    def collect(d):
        for k, v in d.items():
            if torch.is_tensor(v):
                out[k] = v

    collect(data.__dict__)
    store = data.__dict__.get("_store")
    if store is not None:
        collect(store if isinstance(store, dict) else getattr(store, "__dict__", {}))
    return out


def main(in_path, out_path):
    data = _load_data(in_path)
    t = _tensors(data)
    if "pos" not in t:
        sys.exit(f"ERROR: no 'pos' tensor found in {in_path}")

    pos = t["pos"].numpy().astype(np.float32)
    n = pos.shape[0]

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    has_rgb = "f" in t
    has_label = "y" in t
    if has_rgb:
        dtype += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if has_label:
        dtype += [("scalar_cd_type", "f4")]

    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
    if has_rgb:
        rgb = np.clip(t["f"].numpy() * 255.0, 0, 255).round().astype(np.uint8)
        arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if has_label:
        arr["scalar_cd_type"] = t["y"].numpy().astype(np.float32)

    PlyData([PlyElement.describe(arr, "vertex")], byte_order="<").write(out_path)
    print(f"wrote {n:,} points -> {out_path}  (rgb={has_rgb}, label={has_label})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])