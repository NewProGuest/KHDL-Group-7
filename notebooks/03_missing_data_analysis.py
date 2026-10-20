from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import missingno as msno


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUTPUT_PATH = ROOT / "images" / "missing_data_overview.png"


def main():
    df = pd.read_csv(DATA_PATH)

    missing_count = df.isnull().sum()
    missing_pct = df.isnull().mean() * 100

    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct,
    }).sort_values(["missing_count", "missing_pct"], ascending=[False, False])

    print("=== 1. Tổng quan dữ liệu thiếu ===")
    print(summary.head(10))
    print(f"\nTổng số giá trị thiếu trong dataset: {int(summary['missing_count'].sum())}")

    cols_with_missing = summary[summary["missing_count"] > 0]

    if cols_with_missing.empty:
        print("\nKết luận: Dataset gốc không có giá trị thiếu (NaN).")
        print("Do đó, không cần xóa dòng/cột hoặc điền giá trị mean/mode cho bộ dữ liệu này.")
        print("Quá trình EDA và mô hình hóa có thể tiếp tục trên dữ liệu gốc.")
        return

    print("\n=== 2. Các cột có dữ liệu thiếu ===")
    print(cols_with_missing.round(2))

    print("\n=== 3. Phân tích nguyên nhân thiếu dữ liệu ===")
    print("Với dataset IBM HR Analytics gốc, nếu xuất hiện missing values, cần kiểm tra xem đó là MCAR, MAR hay MNAR.")
    print("- MCAR: thiếu hoàn toàn ngẫu nhiên")
    print("- MAR: thiếu phụ thuộc vào các biến khác đã quan sát được")
    print("- MNAR: thiếu phụ thuộc trực tiếp vào giá trị bị thiếu")

    print("\n=== 4. Trực quan hóa missing data ===")
    plt.figure(figsize=(10, 6))
    msno.bar(df, figsize=(10, 6), color="steelblue")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.show()

    plt.figure(figsize=(12, 8))
    msno.matrix(df, figsize=(12, 8), sparkline=False)
    plt.tight_layout()
    plt.show()

    print(f"\nĐã lưu biểu đồ missing data tại: {OUTPUT_PATH}")

    print("\n=== 5. Cách xử lý phù hợp ===")
    print("- Nếu tỷ lệ thiếu < 5%: có thể dùng median/mode hoặc xóa dòng nếu dữ liệu MCAR.")
    print("- Nếu tỷ lệ thiếu > 50%: cân nhắc bỏ cột nếu không quan trọng.")
    print("- Nếu dữ liệu thiếu mang ý nghĩa (MNAR): nên giữ cờ nhị phân hoặc sử dụng kỹ thuật imputation nâng cao.")
    print("- Luôn kiểm tra lại sau xử lý: df.isnull().sum().sum() phải bằng 0 nếu không muốn còn NaN.")


if __name__ == "__main__":
    main()
