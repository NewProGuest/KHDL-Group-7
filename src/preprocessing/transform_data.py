from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "hr_attrition_outliers.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "hr_attrition_transformed.csv"

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
CATEGORICAL_COLUMNS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]
CONSTANT_COLUMNS = ["EmployeeCount", "Over18", "StandardHours"]


def main() -> None:
    data = pd.read_csv(INPUT_PATH)
    required_columns = (
        set(CONTINUOUS_COLUMNS)
        | set(CATEGORICAL_COLUMNS)
        | set(CONSTANT_COLUMNS)
        | {"Attrition", "EmployeeNumber"}
    )
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    attrition = data["Attrition"].map({"No": 0, "Yes": 1})
    if attrition.isna().any():
        unexpected_values = sorted(data.loc[attrition.isna(), "Attrition"].unique())
        raise ValueError(f"Unexpected Attrition values: {unexpected_values}")

    transformed = data.drop(columns=CONSTANT_COLUMNS).copy()
    transformed["Attrition"] = attrition.astype("int64")

    scaler = StandardScaler()
    transformed[CONTINUOUS_COLUMNS] = scaler.fit_transform(
        transformed[CONTINUOUS_COLUMNS]
    )
    transformed = pd.get_dummies(
        transformed,
        columns=CATEGORICAL_COLUMNS,
        dtype="int64",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved transformed data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
