from math import ceil, floor
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "raw" / "hr_attrition_raw.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "hr_attrition_outliers.csv"
REPORT_PATH = ROOT / "data" / "reports" / "outlier_treatment_report.csv"

IQR_MULTIPLIER = 1.5

# Cột dùng để chia nhóm khi tính ngưỡng IQR riêng cho từng nhóm.
GROUP_COLUMN = "JobLevel"
# Nhóm có ít dòng hơn mức này sẽ dùng ngưỡng toàn cục để tránh ngưỡng thiếu ổn định.
MIN_GROUP_SIZE = 30

# Lương phụ thuộc mạnh vào cấp bậc -> tính ngưỡng trong từng JobLevel.
PER_GROUP_COLUMNS = [
    "MonthlyIncome",
]

# Các biến thâm niên lệch phải tự nhiên -> tính ngưỡng trên toàn bộ dữ liệu.
# (Muốn tính theo JobLevel cho biến nào thì chuyển biến đó sang PER_GROUP_COLUMNS.)
GLOBAL_COLUMNS = [
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

# Tất cả các cột trên đều không âm nên cận dưới không được nhỏ hơn 0.
LOWER_FLOOR = 0


def compute_bounds(values: pd.Series) -> dict:
    values = values.dropna()
    if values.empty:
        raise ValueError("Cannot compute IQR bounds without non-null values.")

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = max(LOWER_FLOOR, ceil(q1 - IQR_MULTIPLIER * iqr))
    upper = floor(q3 + IQR_MULTIPLIER * iqr)
    return {"q1": q1, "q3": q3, "iqr": iqr, "lower_bound": lower, "upper_bound": upper}


def build_report_row(column: str, scope: str, n: int, bounds: dict, below: int, above: int) -> dict:
    return {
        "feature": column,
        "scope": scope,
        "method": f"IQR clipping ({IQR_MULTIPLIER} x IQR; integer bounds)",
        "n_rows": n,
        **bounds,
        "values_below": below,
        "values_above": above,
        "values_capped": below + above,
    }


def treat_column(data: pd.DataFrame, column: str, per_group: bool) -> tuple[pd.Series, pd.Series, list[dict]]:
    """Trả về (cột đã clip, cờ 0/1, các dòng báo cáo) cho một cột."""
    original = data[column]
    global_bounds = compute_bounds(original)

    if not per_group:
        lower = pd.Series(global_bounds["lower_bound"], index=data.index)
        upper = pd.Series(global_bounds["upper_bound"], index=data.index)
        below = int((original < lower).sum())
        above = int((original > upper).sum())
        rows = [build_report_row(column, "global", len(data), global_bounds, below, above)]
    else:
        lower = pd.Series(global_bounds["lower_bound"], index=data.index)
        upper = pd.Series(global_bounds["upper_bound"], index=data.index)
        rows = []
        for group_value, group_index in data.groupby(GROUP_COLUMN).groups.items():
            group_values = original.loc[group_index]
            if len(group_values) >= MIN_GROUP_SIZE:
                bounds = compute_bounds(group_values)
                scope = f"{GROUP_COLUMN}={group_value}"
            else:
                bounds = global_bounds
                scope = f"{GROUP_COLUMN}={group_value} (fallback: global bounds)"
            lower.loc[group_index] = bounds["lower_bound"]
            upper.loc[group_index] = bounds["upper_bound"]
            below = int((group_values < bounds["lower_bound"]).sum())
            above = int((group_values > bounds["upper_bound"]).sum())
            rows.append(build_report_row(column, scope, len(group_values), bounds, below, above))

        missing_group_index = data.index[data[GROUP_COLUMN].isna()]
        if not missing_group_index.empty:
            group_values = original.loc[missing_group_index]
            below = int((group_values < global_bounds["lower_bound"]).sum())
            above = int((group_values > global_bounds["upper_bound"]).sum())
            rows.append(
                build_report_row(
                    column,
                    f"{GROUP_COLUMN}=missing (fallback: global bounds)",
                    len(group_values),
                    global_bounds,
                    below,
                    above,
                )
            )

    flag = ((original < lower) | (original > upper)).astype(int)
    clipped = original.clip(lower=lower, upper=upper).astype(original.dtype)
    return clipped, flag, rows


def main() -> None:
    data = pd.read_csv(INPUT_PATH)

    required = set(PER_GROUP_COLUMNS) | set(GLOBAL_COLUMNS) | {GROUP_COLUMN}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    report_rows = []
    flag_columns = []
    treated = data.copy()

    for column, per_group in [(c, True) for c in PER_GROUP_COLUMNS] + [(c, False) for c in GLOBAL_COLUMNS]:
        clipped, flag, rows = treat_column(data, column, per_group)
        treated[column] = clipped
        flag_name = f"{column}_capped"
        treated[flag_name] = flag
        flag_columns.append(flag_name)
        report_rows.extend(rows)

    treated["OutlierFlagCount"] = treated[flag_columns].sum(axis=1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    treated.to_csv(OUTPUT_PATH, index=False)
    pd.DataFrame(report_rows).to_csv(REPORT_PATH, index=False)

    print(f"Saved outlier-treated data to {OUTPUT_PATH}")
    print(f"Saved outlier report to {REPORT_PATH}")
    print(f"Rows with >=1 capped value: {int((treated['OutlierFlagCount'] >= 1).sum())}")
    print(f"Rows with >=2 capped values: {int((treated['OutlierFlagCount'] >= 2).sum())}")


if __name__ == "__main__":
    main()