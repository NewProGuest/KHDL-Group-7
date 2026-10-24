from pathlib import Path
from math import ceil, floor

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "raw" / "hr_attrition_raw.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "hr_attrition_outliers.csv"
REPORT_PATH = ROOT / "data" / "reports" / "outlier_treatment_report.csv"

CONTINUOUS_COLUMNS = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "HourlyRate",
    "MonthlyIncome",
    "MonthlyRate",
    "PercentSalaryHike",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]


def main() -> None:
    data = pd.read_csv(INPUT_PATH)
    missing_columns = set(CONTINUOUS_COLUMNS).difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing continuous columns: {sorted(missing_columns)}")

    report_rows = []
    for column in CONTINUOUS_COLUMNS:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = ceil(q1 - 1.5 * iqr)
        upper_bound = floor(q3 + 1.5 * iqr)
        below = data[column] < lower_bound
        above = data[column] > upper_bound

        report_rows.append(
            {
                "feature": column,
                "method": "IQR clipping (1.5 x IQR; integer bounds)",
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "values_below": int(below.sum()),
                "values_above": int(above.sum()),
                "values_capped": int((below | above).sum()),
            }
        )
        data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    pd.DataFrame(report_rows).to_csv(REPORT_PATH, index=False)
    print(f"Saved outlier-treated data to {OUTPUT_PATH}")
    print(f"Saved outlier report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
