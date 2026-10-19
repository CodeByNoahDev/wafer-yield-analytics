"""End-to-end demo: classify wafer patterns, model yield, run SPC.

Run with:  python -m wafermap            (full run, ~1-2 minutes)
           python -m wafermap --quick    (small run for a smoke test)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import plots
from .classify import compare_models, holdout_evaluation
from .features import FEATURE_NAMES, extract_features_batch
from .generator import DEFAULT_CLASS_COUNTS, make_dataset
from .spc import (
    average_run_length,
    capability,
    ewma,
    first_alarm,
    imr_limits,
    simulate_film_thickness,
    western_electric,
)
from .yield_model import YIELD_MODELS, fit_yield_models, yield_vs_die_area

SPEC_LIMITS_NM = (48.5, 51.5)
PHASE1_RUNS = 50
SPIKE_AT = 75
SHIFT_AT = 100


def run_classification(seed: int, quick: bool, figs: Path) -> dict:
    counts = {k: max(25, v // 6) for k, v in DEFAULT_CLASS_COUNTS.items()} if quick else DEFAULT_CLASS_COUNTS
    wafers, labels = make_dataset(counts, seed=seed)
    X = extract_features_batch(wafers)

    comparison = compare_models(X, labels, seed=seed, folds=3 if quick else 5)
    best = max(comparison, key=lambda name: comparison[name]["macro_f1_mean"])
    holdout = holdout_evaluation(X, labels, model_name=best, seed=seed)

    plots.wafer_gallery(str(figs / "wafer_gallery.png"), seed=seed)
    plots.confusion(str(figs / "confusion_matrix.png"), holdout.confusion, holdout.classes)
    plots.importance(str(figs / "feature_importance.png"), FEATURE_NAMES, holdout.importances, holdout.importance_std)

    per_class = {c: round(holdout.report[c]["f1-score"], 3) for c in holdout.classes}
    top = np.argsort(holdout.importances)[::-1][:5]
    return {
        "n_wafers": len(wafers),
        "n_features": X.shape[1],
        "cv_model_comparison": comparison,
        "selected_model": best,
        "holdout_macro_f1": round(holdout.macro_f1, 4),
        "holdout_accuracy": round(holdout.report["accuracy"], 4),
        "holdout_f1_per_class": per_class,
        "top_features": [FEATURE_NAMES[i] for i in top],
    }


TRUE_D0, TRUE_ALPHA = 0.15, 2.0


def run_yield(seed: int, quick: bool, figs: Path) -> dict:
    sides = np.linspace(3, 28, 14)
    sim = yield_vs_die_area(
        sides, n_wafers=40 if quick else 200, seed=seed, d0_per_cm2=TRUE_D0, alpha=TRUE_ALPHA
    )
    fits = fit_yield_models(sim["area_cm2"], sim["yield"])
    plots.yield_curves(str(figs / "yield_vs_die_area.png"), sim, fits, YIELD_MODELS)
    return {
        "true_d0_per_cm2": TRUE_D0,
        "true_alpha": TRUE_ALPHA,
        "fits": fits,
        "best_model": min(fits, key=lambda k: fits[k]["rmse"]),
        "yield_at_1cm2": float(np.interp(1.0, sim["area_cm2"], sim["yield"])),
    }


def run_spc(seed: int, quick: bool, figs: Path) -> dict:
    x = simulate_film_thickness(seed=seed, spike_at=SPIKE_AT, shift_at=SHIFT_AT)
    limits = imr_limits(x[:PHASE1_RUNS])
    hits = western_electric(x, limits.center, limits.sigma)
    z, ucl, lcl = ewma(x, limits.center, limits.sigma)
    ewma_alarms = np.nonzero((z > ucl) | (z < lcl))[0]
    all_rules = sorted({i for idx in hits.values() for i in idx})

    plots.spc_chart(str(figs / "spc_chart.png"), x, limits, hits, z, ucl, lcl, PHASE1_RUNS, SHIFT_AT)

    def delay(idx):
        first = first_alarm(idx, SHIFT_AT)
        return None if first is None else int(first - SHIFT_AT)

    # Alarms before the shift that are not the planted spike are false alarms.
    false_alarms = [i for i in all_rules if i < SHIFT_AT and i != SPIKE_AT]
    # One chart is one random draw; the fair comparison is the average run
    # length over thousands of simulated runs.
    sims = 300 if quick else 2000
    arl = {
        f"shift_{s}_sigma": average_run_length(s, n_sims=sims, horizon=3000, seed=seed)
        for s in (0.0, 0.5, 1.0, 2.0)
    }
    return {
        "average_run_length": arl,
        "center_nm": round(limits.center, 3),
        "sigma_nm": round(limits.sigma, 3),
        "spike_detected_by_rule1": SPIKE_AT in hits["rule1"],
        "runs_to_detect_shift": {
            "rule1_only": delay(hits["rule1"]),
            "any_western_electric_rule": delay(all_rules),
            "ewma": delay(ewma_alarms),
        },
        "false_alarms_before_shift": false_alarms,
        "capability_before_shift": capability(x[PHASE1_RUNS:SHIFT_AT], *SPEC_LIMITS_NM),
        "capability_after_shift": capability(x[SHIFT_AT:], *SPEC_LIMITS_NM),
    }


def run(out_dir: str = "reports", seed: int = 7, quick: bool = False) -> dict:
    out = Path(out_dir)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    metrics = {
        "classification": run_classification(seed, quick, figs),
        "yield": run_yield(seed, quick, figs),
        "spc": run_spc(seed, quick, figs),
    }
    metrics["runtime_s"] = round(time.perf_counter() - start, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
