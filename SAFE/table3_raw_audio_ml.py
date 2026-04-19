"""
Table 3 (SAFE paper): supervised classifiers on raw digital audio (waveform samples),
10-fold stratified CV, metrics averaged across folds — sklearn all_estimators exploration
(Huang & Garcia, 2023 style).
"""

from __future__ import annotations

import inspect
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import all_estimators
from typing import Any, Callable, Dict, List, Optional, Tuple

# Meta- / composite estimators that need a base estimator or are not suitable here
_SKIP_CLASSIFIERS = frozenset(
    {
        "CalibratedClassifierCV",
        "ClassifierChain",
        "MultiOutputClassifier",
        "OneVsOneClassifier",
        "OneVsRestClassifier",
        "OutputCodeClassifier",
        "SelfTrainingClassifier",
        "StackingClassifier",
        "VotingClassifier",
        "GridSearchCV",
        "RandomizedSearchCV",
        "HalvingGridSearchCV",
        "HalvingRandomSearchCV",
        "SequentialFeatureSelector",
        # Infeasible / prohibitive with ~66k raw samples per clip
        "GaussianProcessClassifier",
    }
)

_GRADIENT_BOOSTING_CLASSIFIERS = frozenset(
    {
        "GradientBoostingClassifier",
        "HistGradientBoostingClassifier",
    }
)

_SCALE_SENSITIVE = frozenset(
    {
        "SVC",
        "NuSVC",
        "LinearSVC",
        "LogisticRegression",
        "SGDClassifier",
        "Perceptron",
        "PassiveAggressiveClassifier",
        "RidgeClassifier",
        "RidgeClassifierCV",
        "KNeighborsClassifier",
        "NearestCentroid",
        "MLPClassifier",
        "LinearDiscriminantAnalysis",
        "QuadraticDiscriminantAnalysis",
    }
)


def raw_waveform_matrix(
    audio_list: List[np.ndarray],
    target_len: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Pad (right) or truncate each clip to fixed length; shape (n_samples, target_len)."""
    rows = np.empty((len(audio_list), target_len), dtype=np.float32)
    for i, a in enumerate(audio_list):
        a = np.asarray(a, dtype=np.float32).ravel()
        if a.size >= target_len:
            rows[i] = a[:target_len]
        else:
            rows[i] = np.pad(a, (0, target_len - a.size), mode="constant", constant_values=pad_value)
    return rows


def _maybe_instantiate(name: str, Est: type, random_state: int) -> Tuple[Optional[Any], Optional[str]]:
    try:
        sig = inspect.signature(Est.__init__)
        params = sig.parameters
        kwargs: Dict[str, Any] = {}
        if "random_state" in params:
            kwargs["random_state"] = random_state
        # cross_validate uses n_jobs=-1; keep estimator n_jobs=1 to avoid oversubscription
        if "n_jobs" in params:
            kwargs["n_jobs"] = 1
        if name == "LinearSVC" and "dual" in params and "max_iter" in params:
            kwargs.setdefault("max_iter", 10000)
        if name in ("SGDClassifier", "PassiveAggressiveClassifier", "Perceptron") and "max_iter" in params:
            kwargs.setdefault("max_iter", 2000)
        if name == "LogisticRegression" and "max_iter" in params:
            kwargs.setdefault("max_iter", 1000)
        if name == "MLPClassifier":
            kwargs.setdefault("max_iter", 200)
            kwargs.setdefault("hidden_layer_sizes", (128,))
        if name == "BernoulliNB" or name == "MultinomialNB":
            # Raw audio is signed; these expect non-negative counts / binary — skip early
            return None, "skipped_non_negative_nb"
        est = Est(**kwargs)
        return est, None
    except TypeError:
        try:
            return Est(), None
        except Exception as e:
            return None, f"init_error:{e!r}"
    except Exception as e:
        return None, f"init_error:{e!r}"


def run_table3_all_classifiers_cv(
    audio_list: List[np.ndarray],
    y: np.ndarray,
    sample_rate: int = 22050,
    raw_duration_sec: float = 3.0,
    n_splits: int = 10,
    random_state: int = 42,
    scale_features: bool = True,
    skip_gradient_boosting: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """
    Run 10-fold stratified CV for each sklearn classifier from all_estimators(type_filter='classifier').
    Rows with NaN scores indicate the estimator failed or was skipped.
    """
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    target_len = max(1, int(round(sample_rate * raw_duration_sec)))
    X = raw_waveform_matrix(audio_list, target_len)
    y = np.asarray(y).astype(np.int64)
    log(f"Raw waveform feature matrix: {X.shape} (target_len={target_len} @ {sample_rate} Hz, ~{raw_duration_sec}s)")
    if skip_gradient_boosting:
        log("Skipping gradient boosting classifiers: " + ", ".join(sorted(_GRADIENT_BOOSTING_CLASSIFIERS)))

    skip_names = set(_SKIP_CLASSIFIERS)
    if skip_gradient_boosting:
        skip_names |= _GRADIENT_BOOSTING_CLASSIFIERS

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average="binary", zero_division=0),
        "recall": make_scorer(recall_score, average="binary", zero_division=0),
        "f1": make_scorer(f1_score, average="binary", zero_division=0),
    }

    rows: List[Dict[str, Any]] = []
    classifiers = all_estimators(type_filter="classifier")

    for name, Est in classifiers:
        if name in skip_names:
            continue
        if not inspect.isclass(Est):
            continue

        est, err = _maybe_instantiate(name, Est, random_state)
        if err == "skipped_non_negative_nb":
            rows.append(
                {
                    "model": name,
                    "feature_type": "raw_waveform",
                    "sampling_rate_hz": int(sample_rate),
                    "accuracy_mean": np.nan,
                    "accuracy_std": np.nan,
                    "precision_mean": np.nan,
                    "precision_std": np.nan,
                    "recall_mean": np.nan,
                    "recall_std": np.nan,
                    "f1_score_mean": np.nan,
                    "f1_score_std": np.nan,
                    "cv_folds": n_splits,
                    "status": err,
                }
            )
            continue
        if est is None:
            rows.append(
                {
                    "model": name,
                    "feature_type": "raw_waveform",
                    "sampling_rate_hz": int(sample_rate),
                    "accuracy_mean": np.nan,
                    "accuracy_std": np.nan,
                    "precision_mean": np.nan,
                    "precision_std": np.nan,
                    "recall_mean": np.nan,
                    "recall_std": np.nan,
                    "f1_score_mean": np.nan,
                    "f1_score_std": np.nan,
                    "cv_folds": n_splits,
                    "status": err or "init_failed",
                }
            )
            continue

        log(f"CV: {name} ...")
        try:
            est_for_cv = clone(est)
            if scale_features and name in _SCALE_SENSITIVE:
                est_for_cv = make_pipeline(StandardScaler(), est_for_cv)
            cv_res = cross_validate(
                est_for_cv,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                error_score=np.nan,
            )
            def stat(key: str) -> Tuple[float, float]:
                vals = cv_res[f"test_{key}"]
                vals = np.asarray(vals, dtype=np.float64)
                if np.all(np.isnan(vals)):
                    return float("nan"), float("nan")
                return float(np.nanmean(vals)), float(np.nanstd(vals))

            am, ast = stat("accuracy")
            pm, pst = stat("precision")
            rm, rst = stat("recall")
            fm, fst = stat("f1")
            status = "ok" if not np.isnan(am) else "cv_failed"
            rows.append(
                {
                    "model": name,
                    "feature_type": "raw_waveform",
                    "sampling_rate_hz": int(sample_rate),
                    "accuracy_mean": am,
                    "accuracy_std": ast,
                    "precision_mean": pm,
                    "precision_std": pst,
                    "recall_mean": rm,
                    "recall_std": rst,
                    "f1_score_mean": fm,
                    "f1_score_std": fst,
                    "cv_folds": n_splits,
                    "status": status,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "model": name,
                    "feature_type": "raw_waveform",
                    "sampling_rate_hz": int(sample_rate),
                    "accuracy_mean": np.nan,
                    "accuracy_std": np.nan,
                    "precision_mean": np.nan,
                    "precision_std": np.nan,
                    "recall_mean": np.nan,
                    "recall_std": np.nan,
                    "f1_score_mean": np.nan,
                    "f1_score_std": np.nan,
                    "cv_folds": n_splits,
                    "status": f"exception:{e!r}",
                }
            )

    return pd.DataFrame(rows)
