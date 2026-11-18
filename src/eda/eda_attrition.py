"""EDA cho IBM HR Attrition: so sánh nhóm nghỉ việc (Yes) và nhóm ở lại (No).

Chạy từ thư mục gốc repo:
    python src/eda/eda_attrition.py

Đầu ra:
    data/reports/eda_numeric_summary.csv     thống kê biến số theo Attrition
    data/reports/eda_attrition_rate.csv      tỷ lệ nghỉ việc theo từng nhóm
    data/reports/figures/eda_*.png           boxplot và biểu đồ cột
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "hr_attrition_raw.csv"
REPORT_DIR = ROOT / "data" / "reports"
FIG_DIR = REPORT_DIR / "figures"

PALETTE = {"No": "#4C78A8", "Yes": "#E4572E"}

NUMERIC_COLS = {
    "MonthlyIncome": "Thu nhập hàng tháng",
    "DistanceFromHome": "Khoảng cách từ nhà đến công ty",
    "Age": "Tuổi",
    "TotalWorkingYears": "Tổng số năm làm việc",
    "YearsAtCompany": "Số năm tại công ty",
    "YearsSinceLastPromotion": "Số năm kể từ lần thăng chức cuối",
    "NumCompaniesWorked": "Số công ty đã làm việc",
    "PercentSalaryHike": "Phần trăm tăng lương",
}

# Biến phân loại / thứ bậc: vẽ tỷ lệ nghỉ việc theo từng mức
RATE_COLS = {
    "OverTime": "Làm thêm giờ",
    "JobSatisfaction": "Mức hài lòng công việc (1-4)",
    "EnvironmentSatisfaction": "Mức hài lòng môi trường (1-4)",
    "WorkLifeBalance": "Cân bằng công việc - cuộc sống (1-4)",
    "BusinessTravel": "Mức độ đi công tác",
    "MaritalStatus": "Tình trạng hôn nhân",
    "Department": "Phòng ban",
    "JobRole": "Vị trí công việc",
    "StockOptionLevel": "Mức quyền chọn cổ phiếu",
}


def attrition_rate_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    grouped = df.groupby(col)["Attrition"].agg(
        n="size", leavers=lambda s: int((s == "Yes").sum())
    )
    grouped["attrition_rate"] = grouped["leavers"] / grouped["n"]
    grouped = grouped.reset_index().rename(columns={col: "group"})
    grouped.insert(0, "feature", col)
    grouped["group"] = grouped["group"].astype(str)
    return grouped


def plot_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (col, label) in zip(axes.ravel(), NUMERIC_COLS.items()):
        sns.boxplot(
            data=df, x="Attrition", y=col, order=["No", "Yes"],
            hue="Attrition", palette=PALETTE, legend=False, ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Ở lại", "Nghỉ việc"])
    fig.suptitle("Nhóm nghỉ việc vs nhóm ở lại (biến số)", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_boxplots_numeric.png", dpi=150)
    plt.close(fig)


def plot_rate_bars(df: pd.DataFrame, overall: float) -> None:
    for col, label in RATE_COLS.items():
        table = attrition_rate_table(df, col)
        if col == "JobRole":
            table = table.sort_values("attrition_rate", ascending=True)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(table["group"], table["attrition_rate"] * 100, color="#E4572E")
            ax.axvline(overall * 100, color="black", ls="--", lw=1)
            ax.set_xlabel("Tỷ lệ nghỉ việc (%)")
            for y, (r, n) in enumerate(zip(table["attrition_rate"], table["n"])):
                ax.text(r * 100 + 0.4, y, f"{r:.0%} (n={n})", va="center", fontsize=9)
        else:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            bars = ax.bar(table["group"], table["attrition_rate"] * 100, color="#E4572E")
            ax.axhline(overall * 100, color="black", ls="--", lw=1)
            ax.set_ylabel("Tỷ lệ nghỉ việc (%)")
            for bar, r, n in zip(bars, table["attrition_rate"], table["n"]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{r:.0%}\n(n={n})", ha="center", fontsize=9)
            ax.set_ylim(0, max(table["attrition_rate"]) * 100 * 1.25)
            plt.setp(ax.get_xticklabels(), rotation=15 if col in {"BusinessTravel", "Department"} else 0)
        ax.set_title(f"Tỷ lệ nghỉ việc theo: {label}  (đường đứt = trung bình {overall:.1%})")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"eda_rate_{col.lower()}.png", dpi=150)
        plt.close(fig)


def plot_income_bands(df: pd.DataFrame, overall: float) -> None:
    bands = pd.cut(df["MonthlyIncome"], [0, 3000, 5000, 8000, 12000, 1e9],
                   labels=["<3k", "3-5k", "5-8k", "8-12k", ">12k"])
    dist = pd.cut(df["DistanceFromHome"], [0, 5, 10, 20, 100],
                  labels=["1-5", "6-10", "11-20", ">20"], include_lowest=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, series, title in (
        (axes[0], bands, "Theo mức thu nhập hàng tháng (USD)"),
        (axes[1], dist, "Theo khoảng cách đến công ty"),
    ):
        rate = df.groupby(series, observed=True)["Attrition"].apply(lambda s: (s == "Yes").mean())
        cnt = df.groupby(series, observed=True).size()
        ax.bar(rate.index.astype(str), rate * 100, color="#E4572E")
        ax.axhline(overall * 100, color="black", ls="--", lw=1)
        for i, (r, n) in enumerate(zip(rate, cnt)):
            ax.text(i, r * 100 + 0.5, f"{r:.0%}\n(n={n})", ha="center", fontsize=9)
        ax.set_ylim(0, rate.max() * 100 * 1.3)
        ax.set_title(title)
        ax.set_ylabel("Tỷ lệ nghỉ việc (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_rate_income_distance_bands.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    overall = (df["Attrition"] == "Yes").mean()

    numeric = (
        df.groupby("Attrition")[list(NUMERIC_COLS)]
        .agg(["mean", "median", "std"])
        .T.reset_index()
    )
    numeric.columns = ["feature", "stat", "stayers_No", "leavers_Yes"]
    numeric.to_csv(REPORT_DIR / "eda_numeric_summary.csv", index=False)

    rates = pd.concat([attrition_rate_table(df, c) for c in RATE_COLS], ignore_index=True)
    rates.to_csv(REPORT_DIR / "eda_attrition_rate.csv", index=False)

    sns.set_theme(style="whitegrid")
    plot_boxplots(df)
    plot_rate_bars(df, overall)
    plot_income_bands(df, overall)

    print(f"Tỷ lệ nghỉ việc chung: {overall:.1%}")
    print(f"Đã lưu bảng EDA vào {REPORT_DIR} và biểu đồ vào {FIG_DIR}")


if __name__ == "__main__":
    main()
