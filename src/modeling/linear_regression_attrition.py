"""Fit a linear probability model and plot employee attrition risk."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "raw" / "hr_attrition_raw.csv"
FIGURE_PATH = ROOT / "data" / "reports" / "figures" / "linear_regression_attrition_risk.png"
PREDICTIONS_PATH = ROOT / "data" / "reports" / "linear_regression_attrition_predictions.csv"
TARGET = "Attrition"
ID_COLUMN = "EmployeeNumber"
RANDOM_STATE = 42


def build_model(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("regression", LinearRegression()),
        ]
    )


def main() -> None:
    data = pd.read_csv(INPUT_PATH)
    required_columns = {TARGET, ID_COLUMN}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if data[TARGET].isna().any():
        raise ValueError(f"{TARGET} contains missing values")

    target = data[TARGET].map({"No": 0, "Yes": 1})
    if target.isna().any():
        unexpected_values = sorted(data.loc[target.isna(), TARGET].unique())
        raise ValueError(f"Unexpected {TARGET} values: {unexpected_values}")

    features = data.drop(columns=[TARGET, ID_COLUMN])
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = features.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable feature columns were found")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        _employee_ids_train,
        employee_ids_test,
    ) = train_test_split(
        features,
        target.astype("int64"),
        data[ID_COLUMN],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = build_model(numeric_columns, categorical_columns)
    model.fit(X_train, y_train)
    raw_predictions = model.predict(X_test)
    probabilities = np.clip(raw_predictions, 0.0, 1.0)

    predictions = pd.DataFrame(
        {
            ID_COLUMN: employee_ids_test.to_numpy(),
            "actual_attrition": y_test.to_numpy(),
            "predicted_attrition_probability": probabilities,
        }
    )
    predictions["risk_decile"] = (
        pd.qcut(
            predictions["predicted_attrition_probability"].rank(method="first"),
            q=min(10, len(predictions)),
            labels=False,
        )
        .astype(int)
        .add(1)
    )
    predictions = predictions.sort_values("risk_decile")
    deciles = (
        predictions.groupby("risk_decile", as_index=False)
        .agg(
            predicted_risk=("predicted_attrition_probability", "mean"),
            observed_attrition_rate=("actual_attrition", "mean"),
            employees=(ID_COLUMN, "count"),
        )
    )

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        deciles["risk_decile"],
        deciles["predicted_risk"] * 100,
        marker="o",
        linewidth=2,
        label="Khả năng nghỉ việc dự đoán",
    )
    ax.plot(
        deciles["risk_decile"],
        deciles["observed_attrition_rate"] * 100,
        marker="s",
        linewidth=2,
        label="Tỷ lệ nghỉ việc thực tế",
    )
    for row in deciles.itertuples(index=False):
        ax.annotate(
            f"n={row.employees}",
            (row.risk_decile, row.predicted_risk * 100),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set(
        title="Khả năng nghỉ việc theo nhóm rủi ro\nLinear Regression (tập kiểm tra)",
        xlabel="Nhóm rủi ro (1 = thấp nhất, 10 = cao nhất)",
        ylabel="Tỷ lệ / khả năng nghỉ việc (%)",
        xticks=deciles["risk_decile"],
        ylim=(0, 100),
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=160)
    plt.close(fig)

    rmse = mean_squared_error(y_test, probabilities) ** 0.5
    r2 = r2_score(y_test, probabilities)
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test R²: {r2:.3f}")
    print(f"Saved chart to {FIGURE_PATH}")
    print(f"Saved test predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
