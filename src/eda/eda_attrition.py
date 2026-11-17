"""EDA for IBM HR Attrition.

This script produces summary tables and charts for leavers vs stayers.
It is intended to be run after data quality and outlier handling.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "hr_attrition_raw.csv"
REPORT_DIR = ROOT / "data" / "reports"
FIG_DIR = REPORT_DIR / "figures"
OUT_SUMMARY = REPORT_DIR / "eda_summary.csv"


def save_chart(df: pd.DataFrame, column: str, title: str, save_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if df[column].dtype == "object" or df[column].dtype.name == "category":
        counts = df.groupby(["Attrition", column]).size().unstack(fill_value=0)
        counts.plot(kind="bar", ax=ax)
        ax.set_ylabel("Số lượng")
    else:
        by_attrition = df.groupby("Attrition")[column].describe()
        by_attrition.plot(kind="bar", ax=ax)
        ax.set_ylabel(column)
    ax.set_title(title)
    ax.legend(title="Attrition")
    fig.tight_layout()
    fig.savefig(FIG_DIR / save_name, dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if "Attrition" not in df.columns:
        raise ValueError("Attrition column not found in raw data")

    # Basic summary table for quick exploration.
    summary = []
    for col in df.columns:
        if col == "Attrition":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            stats = df.groupby("Attrition")[col].agg(["mean", "median", "std", "min", "max"])
            for value, row in stats.iterrows():
                summary.append({
                    "feature": col,
                    "attrition": value,
                    "mean": row["mean"],
                    "median": row["median"],
                    "std": row["std"],
                    "min": row["min"],
                    "max": row["max"],
                })
        else:
            table = df.groupby(["Attrition", col]).size().reset_index(name="count")
            for _, row in table.iterrows():
                summary.append({
                    "feature": col,
                    "attrition": row["Attrition"],
                    "category": row[col],
                    "count": int(row["count"]),
                })

    pd.DataFrame(summary).to_csv(OUT_SUMMARY, index=False)

    # Core EDA charts.
    for col in [
        "Age",
        "MonthlyIncome",
        "TotalWorkingYears",
        "YearsAtCompany",
        "DistanceFromHome",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "OverTime",
        "Department",
        "JobRole",
    ]:
        if col in df.columns:
            save_chart(df, col, f"Attrition vs {col}", f"eda_{col.lower()}.png")

    # Attrition rate by department and overtime.
    attrition_rate = df.groupby("Department")["Attrition"].apply(
        lambda s: (s == "Yes").mean()
    ).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    attrition_rate.plot(kind="bar", color="tomato", ax=ax)
    ax.set_title("Tỷ lệ nghỉ việc theo Department")
    ax.set_ylabel("Tỷ lệ nghỉ việc")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_attrition_by_department.png", dpi=150)
    plt.close(fig)

    overtime_rate = df.groupby("OverTime")["Attrition"].apply(
        lambda s: (s == "Yes").mean()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    overtime_rate.plot(kind="bar", color="steelblue", ax=ax)
    ax.set_title("Tỷ lệ nghỉ việc theo OverTime")
    ax.set_ylabel("Tỷ lệ nghỉ việc")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_attrition_by_overtime.png", dpi=150)
    plt.close(fig)

    print(f"Saved EDA summary to {OUT_SUMMARY}")
    print(f"Saved EDA charts to {FIG_DIR}")


if __name__ == "__main__":
    main()
