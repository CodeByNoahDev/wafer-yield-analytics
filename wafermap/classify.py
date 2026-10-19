"""Train and evaluate wafer-pattern classifiers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def candidate_models(seed: int = 0) -> dict[str, object]:
    """A simple baseline plus two tree ensembles.

    class_weight="balanced" stops the models from just predicting "none" for
    everything - with imbalanced fab data that would still score ~40% accuracy.
    """
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            early_stopping=True,
            n_iter_no_change=15,
            class_weight="balanced",
            random_state=seed,
        ),
    }


def compare_models(X: np.ndarray, y: np.ndarray, seed: int = 0, folds: int = 5) -> dict[str, dict[str, float]]:
    """Stratified k-fold cross-validated macro-F1 for every candidate model.

    Macro-F1 averages the F1 score of each class equally, so a rare class like
    near_full counts as much as the common "none" class.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results = {}
    for name, model in candidate_models(seed).items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
        results[name] = {"macro_f1_mean": float(scores.mean()), "macro_f1_std": float(scores.std())}
    return results


@dataclass
class HoldoutResult:
    model: object
    classes: np.ndarray
    macro_f1: float
    report: dict
    confusion: np.ndarray
    importances: np.ndarray  # permutation importance, mean drop in macro-F1
    importance_std: np.ndarray


def holdout_evaluation(
    X: np.ndarray, y: np.ndarray, model_name: str = "random_forest", seed: int = 0, test_size: float = 0.25
) -> HoldoutResult:
    """Fit on a training split and report metrics on an untouched test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    model = candidate_models(seed)[model_name]
    model.fit(X_train, y_train)
    if isinstance(model, RandomForestClassifier):
        # Parallel fitting helps, but for the hundreds of small predict() calls
        # in permutation importance the thread start-up cost dominates.
        model.set_params(n_jobs=1)
    pred = model.predict(X_test)
    classes = np.unique(y)

    # Permutation importance: shuffle one feature at a time and measure how much
    # the test score drops. Unlike tree "impurity" importance it is measured on
    # unseen data and is not biased towards high-variance features.
    perm = permutation_importance(
        model, X_test, y_test, scoring="f1_macro", n_repeats=5, random_state=seed, n_jobs=1
    )
    return HoldoutResult(
        model=model,
        classes=classes,
        macro_f1=float(f1_score(y_test, pred, average="macro")),
        report=classification_report(y_test, pred, output_dict=True, zero_division=0),
        confusion=confusion_matrix(y_test, pred, labels=classes),
        importances=perm.importances_mean,
        importance_std=perm.importances_std,
    )
