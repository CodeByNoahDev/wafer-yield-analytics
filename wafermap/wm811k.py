"""Optional loader for the real WM-811K dataset (LSWMD.pkl).

WM-811K holds 811,457 wafer maps from a real fab, about 172k of them labelled
(Wu, Jang & Chen, 2015). It is distributed as a pandas pickle, e.g. on Kaggle
("WM-811K wafer map"). Download it yourself and pass the path to load_wm811k().

NOTE: the pickle was written with a very old pandas version, so it needs the
module alias below to load on modern pandas. Only unpickle files from a source
you trust - pickle can run code when loaded.
"""
from __future__ import annotations

import sys

import numpy as np

LABEL_MAP = {
    "Center": "center",
    "Donut": "donut",
    "Edge-Loc": "edge_loc",
    "Edge-Ring": "edge_ring",
    "Loc": "loc",
    "Random": "random",
    "Scratch": "scratch",
    "Near-full": "near_full",
    "none": "none",
}


def _label(failure_type) -> str | None:
    # failureType is stored as a nested array like [['Center']], or [] if unlabelled.
    arr = np.asarray(failure_type).ravel()
    return LABEL_MAP.get(str(arr[0])) if arr.size else None


def load_wm811k(path: str, max_per_class: int | None = None, seed: int = 0):
    """Return (wafers, labels) for the labelled WM-811K maps."""
    import pandas as pd
    import pandas.core.indexes as indexes

    sys.modules.setdefault("pandas.indexes", indexes)  # old-pandas compatibility
    df = pd.read_pickle(path)
    df["label"] = df["failureType"].apply(_label)
    df = df[df["label"].notna()]
    if max_per_class is not None:
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max_per_class), random_state=seed))
        )
    wafers = [np.asarray(w, dtype=np.int8) for w in df["waferMap"]]
    return wafers, df["label"].to_numpy()
