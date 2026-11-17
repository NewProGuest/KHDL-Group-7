"""Train attrition models with proper train/test pipeline and threshold tuning.

This script avoids data leakage by fitting preprocessing inside the pipeline on the
training set only. It saves a comparison table plus the list of high-risk employees.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:  # pragma: no cover
    SMOTE = None
    ImbPipeline = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "hr_attrition_raw.csv"
REPORT_DIR = ROOT / "data" / "reports"
FIG_DIR = REPORT_DIR / "figures"
MODEL_REPORT = REPORT_DIR / "model_comparison.csv"
HIGH_RISK_PATH = REPORT_DIR / "high_risk_employees.csv"
SEED = 42


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def make_models() -> dict:
    models = {
        "LogReg_balanced": LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        ),
        "RandomForest_balanced": __import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            scale_pos_weight=5.0,
        )
    return models


def fit_and_evaluate(model_name: str, estimator, X_train, y_train, X_val, y_val, X_test, y_test):
    if SMOTE is not None and model_name in {"LogReg_balanced", "XGBoost"}:
        pipeline = ImbPipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train)),
                ("smote", SMOTE(random_state=SEED, k_neighbors=3)),
                ("model", estimator),
            ]
        )
    else:
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )

    pipeline.fit(X_train, y_train)
    val_proba = pipeline.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.30, 0.70, 9):
        pred = (val_proba >= threshold).astype(int)
        score = f1_score(y_val, pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_threshold).astype(int)

    metrics = {
        "model": model_name,
        "threshold": float(best_threshold),
        "accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "recall": recall_score(y_test, test_pred, zero_division=0),
        "f1": f1_score(y_test, test_pred, zero_division=0),
        "auc": roc_auc_score(y_test, test_proba),
    }
    return pipeline, metrics, test_proba, test_pred, best_threshold


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    if "Attrition" not in df.columns:
        raise ValueError("Missing Attrition target")

    X = df.drop(columns=["Attrition", "EmployeeNumber"])
    y = df["Attrition"].map({"No": 0, "Yes": 1}).astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    rows = []
    model_outputs = {}
    for model_name, estimator in make_models().items():
        pipeline, metrics, test_proba, test_pred, threshold = fit_and_evaluate(
            model_name, estimator, X_train, y_train, X_val, y_val, X_test, y_test
        )
        rows.append(metrics)
        model_outputs[model_name] = {
            "pipeline": pipeline,
            "proba": test_proba,
            "pred": test_pred,
            "threshold": threshold,
            "y_true": y_test,
            "X_test": X_test,
        }

        cm = confusion_matrix(y_test, test_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{model_name} confusion matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No", "Yes"])
        ax.set_yticklabels(["No", "Yes"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"confusion_matrix_{model_name.lower()}.png", dpi=150)
        plt.close(fig)

    pd.DataFrame(rows).to_csv(MODEL_REPORT, index=False)

    final_name = "LogReg_balanced"
    final_model = model_outputs[final_name]["pipeline"]
    final_X = X_test.copy()
    final_y = y_test.copy()
    final_proba = final_model.predict_proba(final_X)[:, 1]
    final_pred = (final_proba >= model_outputs[final_name]["threshold"]).astype(int)

    risk_df = X_test.copy()
    risk_df["EmployeeNumber"] = df.loc[X_test.index, "EmployeeNumber"].values
    risk_df["actual_attrition"] = final_y.values
    risk_df["predicted_probability"] = final_proba
    risk_df["predicted_attrition"] = final_pred
    risk_df = risk_df.sort_values("predicted_probability", ascending=False)
    risk_df = risk_df[risk_df["predicted_attrition"] == 1].head(100)
    risk_df.to_csv(HIGH_RISK_PATH, index=False)

    print(f"Saved model comparison to {MODEL_REPORT}")
    print(f"Saved high-risk employees to {HIGH_RISK_PATH}")

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
