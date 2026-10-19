"""Figures for the README / report."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .generator import PATTERNS, generate_wafer
from .spc import RULES

WAFER_CMAP = ListedColormap(["#ffffff", "#cfe3d4", "#c0392b"])  # off-wafer, pass, fail


def wafer_gallery(path: str, seed: int = 1, per_class: int = 3) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(per_class, len(PATTERNS), figsize=(2 * len(PATTERNS), 2.1 * per_class))
    for col, pattern in enumerate(PATTERNS):
        for row in range(per_class):
            ax = axes[row, col]
            ax.imshow(generate_wafer(pattern, size=44, rng=rng), cmap=WAFER_CMAP, vmin=0, vmax=2)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(pattern, fontsize=11)
    fig.suptitle("Synthetic wafer maps by defect pattern (red = failing die)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def confusion(path: str, cm: np.ndarray, classes: np.ndarray) -> None:
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if norm[i, j] > 0.6 else "black", fontsize=9)
    ax.set_xlabel("Predicted pattern")
    ax.set_ylabel("True pattern")
    ax.set_title("Hold-out confusion matrix (counts, coloured by row %)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def importance(path: str, names: list[str], mean: np.ndarray, std: np.ndarray, top: int = 15) -> None:
    order = np.argsort(mean)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh([names[i] for i in order], mean[order], xerr=std[order], color="#2e86c1")
    ax.set_xlabel("Drop in macro-F1 when the feature is shuffled")
    ax.set_title(f"Top {top} features (permutation importance)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def yield_curves(path: str, sim: dict, fits: dict, model_fns: dict) -> None:
    area = sim["area_cm2"]
    grid = np.linspace(0, area.max() * 1.05, 200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax1.plot(area, sim["yield"], "ko", label="Monte Carlo (density varies wafer to wafer)")
    for name, fit in fits.items():
        ax1.plot(grid, model_fns[name](grid, *fit["params"]), label=f"{name} (RMSE {fit['rmse']:.3f})")
    ax1.set_xlabel("Die area (cm$^2$)")
    ax1.set_ylabel("Die yield")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("Yield vs die area: model fits")
    ax1.legend(fontsize=8)

    good = sim["dies_per_wafer"] * sim["yield"]
    ax2.plot(area, sim["dies_per_wafer"], "s--", color="grey", label="Gross dies per wafer")
    ax2.plot(area, good, "o-", color="#c0392b", label="Good dies per wafer")
    ax2.set_yscale("log")
    ax2.set_xlabel("Die area (cm$^2$)")
    ax2.set_title("Why big chips are expensive (and chiplets help)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def spc_chart(path: str, x: np.ndarray, limits, hits: dict, ewma_z, ewma_ucl, ewma_lcl,
              phase1_end: int, shift_at: int) -> None:
    n = np.arange(len(x))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(n, x, "-o", ms=3, color="#34495e", lw=0.8)
    for k, style in ((3, "-"), (2, ":"), (1, ":")):
        for sign in (1, -1):
            ax1.axhline(limits.center + sign * k * limits.sigma, color="#c0392b" if k == 3 else "#999",
                        ls=style, lw=1)
    ax1.axhline(limits.center, color="green", lw=1)
    markers = {"rule1": "X", "rule2": "D", "rule3": "s", "rule4": "^", "trend": "v"}
    for rule, idx in hits.items():
        if idx:
            ax1.scatter(idx, x[idx], marker=markers[rule], s=60, zorder=3, label=RULES[rule])
    ax1.axvspan(0, phase1_end, color="#eaf2f8", label="Phase I (limits set here)")
    ax1.axvline(shift_at, color="orange", ls="--", label="True +1 sigma shift")
    ax1.set_ylabel("Film thickness (nm)")
    ax1.set_title("Individuals chart with Western Electric rules")
    ax1.legend(fontsize=7, loc="upper left", ncol=2)

    ax2.plot(n, ewma_z, "-", color="#2e86c1", label="EWMA (lambda = 0.2)")
    ax2.plot(n, ewma_ucl, "r-", lw=1)
    ax2.plot(n, ewma_lcl, "r-", lw=1)
    out = (ewma_z > ewma_ucl) | (ewma_z < ewma_lcl)
    ax2.scatter(n[out], ewma_z[out], color="red", zorder=3, s=20, label="EWMA alarm")
    ax2.axvline(shift_at, color="orange", ls="--")
    ax2.set_xlabel("Run number")
    ax2.set_ylabel("EWMA (nm)")
    ax2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
