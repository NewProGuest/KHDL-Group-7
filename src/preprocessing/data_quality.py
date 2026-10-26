import json
import re
import pandas as pd
from pathlib import Path

# Xác định thư mục gốc dự án
ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "raw" / "hr_attrition_raw.csv"

print("Đường dẫn dữ liệu:", DATA_FILE)
print("File tồn tại:", DATA_FILE.exists())

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Không tìm thấy file: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print("\n===== DATA QUALITY REPORT =====")
print("Kích thước dữ liệu:", df.shape)

print("\n1. Kiểu dữ liệu:")
print(df.dtypes)

print("\n2. Giá trị thiếu:")
print(df.isnull().sum())

print("\n3. Số dòng trùng lặp:", int(df.duplicated().sum()))

print("\n4. Thống kê dữ liệu số:")
print(df.describe().T)

print("\n5. Số giá trị phân biệt:")
print(df.nunique())

# ==========================================
# 1. KIỂM TRA GIÁ TRỊ BẤT HỢP LỆ
# ==========================================

print("\n===== KIỂM TRA GIA TRI BAT THUONG =====")

non_negative_cols = [
    "Age", "DailyRate", "DistanceFromHome",
    "HourlyRate", "MonthlyIncome", "MonthlyRate",
    "TotalWorkingYears", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager", "NumCompaniesWorked"
]

negative_counts = {}
for col in non_negative_cols:
    count = int((df[col] < 0).sum())
    negative_counts[col] = count
    print(f"{col}: {count} gia tri am")

salary_invalid = int((df["MonthlyIncome"] <= 0).sum())
print("\nLuong thang <= 0:", salary_invalid)

# Kiểm tra quan hệ giữa các cột số năm làm việc
checks = {
    "YearsAtCompany > TotalWorkingYears":
        df["YearsAtCompany"] > df["TotalWorkingYears"],
    "YearsInCurrentRole > YearsAtCompany":
        df["YearsInCurrentRole"] > df["YearsAtCompany"],
    "YearsWithCurrManager > YearsAtCompany":
        df["YearsWithCurrManager"] > df["YearsAtCompany"],
    "YearsSinceLastPromotion > YearsAtCompany":
        df["YearsSinceLastPromotion"] > df["YearsAtCompany"],
}

consistency_counts = {}
for name, condition in checks.items():
    count = int(condition.sum())
    consistency_counts[name] = count
    print(f"{name}: {count} dong")

# Kiểm tra các giá trị phân loại hợp lệ
expected_values = {
    "Attrition": ["Yes", "No"],
    "OverTime": ["Yes", "No"],
    "Gender": ["Male", "Female"],
    "Over18": ["Y"],
}

invalid_categories = {}
for col, allowed in expected_values.items():
    count = int((~df[col].isin(allowed)).sum())
    invalid_categories[col] = count
    print(f"{col} - gia tri khong hop le: {count}")

# Kiểm tra cột chỉ có một giá trị
constant_cols = [
    col for col in df.columns
    if df[col].nunique(dropna=False) <= 1
]
print("\nCot chi co mot gia tri:", constant_cols)

# ==========================================
# 2. PHÁT HIỆN GIÁ TRỊ NGOẠI LỆ (IQR)
# ==========================================

print("\n===== OUTLIER CHECK (IQR) =====")

# Chỉ xét các biến số có ý nghĩa để kiểm tra ngoại lệ.
# Không tự động xóa các giá trị bị phát hiện.
outlier_cols = [
    "Age", "DailyRate", "DistanceFromHome", "HourlyRate",
    "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "PercentSalaryHike", "TotalWorkingYears",
    "TrainingTimesLastYear", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]

outlier_results = {}

for col in outlier_cols:
    series = df[col].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((series < lower) | (series > upper)).sum())

    outlier_results[col] = {
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "outlier_count": count
    }

    print(f"{col}: {count} outliers "
          f"(khoang tham chieu: {lower:.2f} - {upper:.2f})")

# ==========================================
# 3. KIỂM TRA EMOJI VÀ KÝ TỰ ĐIỀU KHIỂN
# ==========================================

print("\n===== TEXT QUALITY CHECK =====")

text_cols = df.select_dtypes(include=["object", "str"]).columns

emoji_pattern = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE
)

text_quality_results = {}

for col in text_cols:
    values = df[col].dropna().astype(str)

    emoji_rows = int(values.apply(
        lambda value: bool(emoji_pattern.search(value))
    ).sum())

    control_rows = int(values.apply(
        lambda value: any(
            ord(char) < 32 and char not in "\t\n\r"
            for char in value
        )
    ).sum())

    text_quality_results[col] = {
        "rows_with_emoji": emoji_rows,
        "rows_with_control_characters": control_rows
    }

    print(
        f"{col}: emoji = {emoji_rows}, "
        f"control characters = {control_rows}"
    )

# ==========================================
# 4. LÀM SẠCH CƠ BẢN VÀ LƯU FILE
# ==========================================

# Tạo bản sao để bảo toàn dữ liệu gốc
clean_df = df.copy()

# Chuẩn hóa khoảng trắng ở đầu và cuối chuỗi
clean_text_cols = clean_df.select_dtypes(
    include=["object", "str"]
).columns

for col in clean_text_cols:
    clean_df[col] = clean_df[col].str.strip()

# Đếm thay đổi do loại bỏ khoảng trắng
whitespace_changes = {}
for col in clean_text_cols:
    before = df[col].dropna().astype(str)
    after = clean_df.loc[before.index, col].astype(str)
    whitespace_changes[col] = int((before != after).sum())

print("\n===== CLEANING SUMMARY =====")
print("So gia tri van ban da duoc chuan hoa khoang trang:",
      sum(whitespace_changes.values()))
print("Khong tu dong xoa emoji, ky tu la hoac outliers.")

# Tạo thư mục đầu ra
output_dir = ROOT / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

clean_file = output_dir / "hr_attrition_clean.csv"
clean_df.to_csv(clean_file, index=False, encoding="utf-8-sig")

# Báo cáo chất lượng đầy đủ hơn
quality_report = {
    "rows_before": int(len(df)),
    "columns_before": int(len(df.columns)),
    "missing_values_before": int(df.isnull().sum().sum()),
    "duplicate_rows_before": int(df.duplicated().sum()),
    "rows_after": int(len(clean_df)),
    "columns_after": int(len(clean_df.columns)),
    "missing_values_after": int(clean_df.isnull().sum().sum()),
    "duplicate_rows_after": int(clean_df.duplicated().sum()),
    "negative_value_counts": negative_counts,
    "monthly_income_non_positive": salary_invalid,
    "working_year_consistency_violations": consistency_counts,
    "invalid_category_counts": invalid_categories,
    "constant_columns": constant_cols,
    "outlier_check_iqr": outlier_results,
    "text_quality_check": text_quality_results,
    "whitespace_changes_by_column": whitespace_changes,
    "monthly_income_dtype": str(clean_df["MonthlyIncome"].dtype),
    "emoji_or_control_characters_removed": False,
    "outliers_removed": False
}

report_file = output_dir / "data_quality_report.json"
with open(report_file, "w", encoding="utf-8") as file:
    json.dump(quality_report, file, ensure_ascii=False, indent=4)

print("\n===== KET QUA LAM SACH =====")
print("File du lieu sach:", clean_file)
print("File bao cao:", report_file)
print(json.dumps(quality_report, ensure_ascii=False, indent=4))



