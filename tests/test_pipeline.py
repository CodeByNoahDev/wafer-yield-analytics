from wafermap.pipeline import run


def test_quick_pipeline_runs_end_to_end(tmp_path):
    metrics = run(str(tmp_path), seed=0, quick=True)
    figures = {p.name for p in (tmp_path / "figures").iterdir()}
    assert {"wafer_gallery.png", "confusion_matrix.png", "feature_importance.png",
            "yield_vs_die_area.png", "spc_chart.png"} <= figures
    assert (tmp_path / "metrics.json").exists()
    assert metrics["classification"]["holdout_macro_f1"] > 0.7
    assert metrics["spc"]["spike_detected_by_rule1"]
