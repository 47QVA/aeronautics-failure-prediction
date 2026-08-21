"""Train baseline gradient-boosting models for both modeling framings:

  - RUL regression:              XGBRegressor, target = RUL (cycles)
  - Binary failure classification: XGBClassifier, target = RUL <= FAILURE_HORIZON

Unit-level split (src/splits.py): train on TRAIN_UNITS, model-select against
VAL_UNITS, report final numbers on the held-out TEST_UNITS (NASA's own
unit-disjoint test split, touched only here).

Usage:
    .venv\\Scripts\\python.exe src\\train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from data_loading import load_cycle_level
from evaluate import classification_report_dict, regression_report, select_threshold
from features import add_failure_label, build_features, split_xy
from splits import TEST_UNITS, TRAIN_UNITS, VAL_UNITS

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
REPORTS_PATH = REPO_ROOT / "data" / "processed" / "phase3_metrics.json"


def _prepare() -> pd.DataFrame:
    dev = load_cycle_level("dev")
    test = load_cycle_level("test")
    df = pd.concat([dev, test], ignore_index=True)
    df = build_features(df)
    df = add_failure_label(df)
    return df


def _role(unit: int) -> str:
    if unit in TRAIN_UNITS:
        return "train"
    if unit in VAL_UNITS:
        return "val"
    if unit in TEST_UNITS:
        return "test"
    raise ValueError(f"unit {unit} not in any split")


def train_rul_model(df: pd.DataFrame) -> tuple[XGBRegressor, dict]:
    X, y, _ = split_xy(df)
    role = df["unit"].map(_role)

    model = XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        early_stopping_rounds=20,
        random_state=0,
    )
    model.fit(
        X[role == "train"], y[role == "train"],
        eval_set=[(X[role == "val"], y[role == "val"])],
        verbose=False,
    )

    reports = {}
    for split in ("train", "val", "test"):
        mask = role == split
        pred = model.predict(X[mask])
        reports[split] = regression_report(y[mask], pred)
    return model, reports


def train_classifier(df: pd.DataFrame) -> tuple[XGBClassifier, dict]:
    X, _, y = split_xy(df)
    role = df["unit"].map(_role)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        early_stopping_rounds=20,
        random_state=0,
        eval_metric="logloss",
    )
    model.fit(
        X[role == "train"], y[role == "train"],
        eval_set=[(X[role == "val"], y[role == "val"])],
        verbose=False,
    )

    val_mask = role == "val"
    val_proba = model.predict_proba(X[val_mask])[:, 1]
    threshold = select_threshold(y[val_mask], val_proba)

    reports = {"decision_threshold": threshold}
    for split in ("train", "val", "test"):
        mask = role == split
        proba = model.predict_proba(X[mask])[:, 1]
        pred = (proba >= threshold).astype(int)
        reports[split] = classification_report_dict(y[mask], pred, proba)
    return model, reports


def main() -> None:
    df = _prepare()

    rul_model, rul_reports = train_rul_model(df)
    clf_model, clf_reports = train_classifier(df)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(rul_model, MODELS_DIR / "rul_regressor.joblib")
    joblib.dump(clf_model, MODELS_DIR / "failure_classifier.joblib")

    metrics = {
        "rul_regression": {"best_iteration": int(rul_model.best_iteration), **rul_reports},
        "failure_classification": {"best_iteration": int(clf_model.best_iteration), **clf_reports},
    }
    REPORTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"RUL regression (best_iteration={rul_model.best_iteration}):")
    print(pd.DataFrame(rul_reports).T)
    print()
    print(f"Failure classification (best_iteration={clf_model.best_iteration}):")
    for split, r in clf_reports.items():
        print(split, r)


if __name__ == "__main__":
    main()
