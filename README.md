# IBM HR Analytics: Employee Attrition Prediction

> Đồ án Khoa học Dữ liệu - Nhóm 7  
> Phân tích nguyên nhân nghỉ việc của nhân sự tại IBM HR Analytics Employee Attrition & Performance.

## 1. Tóm tắt dự án

Dự án phân tích dữ liệu nhân sự để xác định các nhân tố ảnh hưởng đến tỷ lệ nghỉ việc (`Attrition`) và xây dựng mô hình dự báo khả năng rời công ty. Mục tiêu không chỉ là độ chính xác của mô hình mà còn là hỗ trợ phòng nhân sự (HR) đưa ra các quyết định can thiệp và giữ chân nhân tài.

Bộ dữ liệu gốc là IBM HR Analytics Employee Attrition & Performance từ Kaggle với 1.470 nhân viên, 35 cột, và biến mục tiêu `Attrition` (Yes/No). Tỷ lệ nghỉ việc thực tế khoảng 16.1% (Yes) và 83.9% (No), nên mô hình cần xử lý mất cân bằng lớp.

## 2. Cấu trúc kho dữ liệu

```text
KHDL-Group-7/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
├── docs/
├── notebooks/
├── src/
│   ├── eda/
│   ├── modeling/
│   ├── preprocessing/
│   └── reduction/
├── models/
└── .github/            (nếu cần thêm CI sau này)
```

## 3. Thư mục và script chính

- `data/raw/`: dữ liệu gốc, không chỉnh sửa.
- `data/processed/`: dữ liệu đã làm sạch, xử lý outlier, biến đổi mô hình.
- `data/reports/`: bảng báo cáo, biểu đồ EDA, mô hình, và danh sách nhân viên rủi ro cao.
- `docs/`: tài liệu phân tích, overview, và khuyến nghị giữ chân nhân sự.
- `notebooks/`: phân tích EDA dạng notebook.
- `src/eda/`: script khám phá dữ liệu (EDA).
- `src/preprocessing/`: xử lý dữ liệu thiếu, outlier và chuẩn hóa.
- `src/reduction/`: feature selection và so sánh phương án giảm chiều.
- `src/modeling/`: huấn luyện mô hình phân loại và xuất bản dự đoán.
- `models/`: mô hình đã huấn luyện hoặc metadata mô hình.

## 4. Quy trình chạy dự án

Chạy từ thư mục gốc repo:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/preprocessing/data_quality.py
python src/preprocessing/handle_outliers.py
python src/preprocessing/transform_data.py
python src/eda/eda_attrition.py
python src/reduction/feature_selection.py
python src/modeling/train_attrition_models.py
```

## 5. Các output quan trọng

Sau khi chạy các script, repo sẽ tạo ra các file sau:

- `data/processed/hr_attrition_clean.csv`
- `data/processed/hr_attrition_outliers.csv`
- `data/processed/hr_attrition_transformed.csv`
- `data/processed/hr_attrition_reduced.csv`
- `data/reports/eda_summary.csv`
- `data/reports/high_risk_employees.csv`
- `data/reports/model_comparison.csv`
- `data/reports/figures/*.png`
- `docs/retention_recommendations.md`

## 6. Mô hình và xử lý mất cân bằng lớp

Mô hình cuối cùng sử dụng pipeline chuẩn hóa/encoder và mô hình phân loại thực tế với xử lý mất cân bằng lớp:

- Logistic Regression với `class_weight='balanced'`
- Random Forest với `class_weight='balanced_subsample'`
- XGBoost nếu cài đặt sẵn
- SMOTE trong pipeline khi `imbalanced-learn` có sẵn
- tuning ngưỡng (`threshold tuning`) trên tập validation để tối ưu F1 cho lớp Attrition = Yes

Điều này quan trọng vì tỷ lệ nghỉ việc khoảng 16%, nên không nên chỉ dựa trên accuracy thuần túy.

## 7. Khuyến nghị giữ chân nhân sự (HR retention recommendations)

Những yếu tố xuất hiện thường xuyên trong EDA và mô hình là nền tảng cho các khuyến nghị sau:

- Cảnh báo sớm cho nhân viên có khoảng thời gian làm việc ngắn và mức độ hài lòng thấp.
- Theo dõi sự thay đổi trong `OverTime`, `JobSatisfaction`, `EnvironmentSatisfaction`, `WorkLifeBalance`.
- Can thiệp cho nhân viên ở cụm `JobRole` có tỷ lệ nghỉ việc cao, đặc biệt là vai trò có áp lực cao.
- Xem xét mức lương và quyền lợi khi `MonthlyIncome`, `StockOptionLevel`, `PercentSalaryHike` thấp so với peer group.
- Tăng hoạt động hỗ trợ quản lý, đánh giá hiệu suất và định hướng nghề nghiệp để giảm rủi ro nghỉ việc.

Chi tiết đầy đủ: `docs/retention_recommendations.md`.

## 8. EDA và phân tích nguyên nhân

Script EDA thu thập các chỉ số sau:

- Tỷ lệ nghỉ việc theo phòng ban, giới tính, overtime, và vai trò công việc.
- Phân phối độ tuổi / thâm niên / khoảng cách từ nhà / thu nhập.
- So sánh leavers vs stayers theo các chỉ số cảm nhận và đánh giá.
- Phát hiện các biến có mức độ liên quan mạnh với Attrition.

Script: `src/eda/eda_attrition.py`

## 9. Dữ liệu và mô hình hóa

- `src/preprocessing/*.py`: kiểm tra dữ liệu, xử lý outlier, biến đổi.
- `src/reduction/feature_selection.py`: chọn và so sánh đặc trưng.
- `src/modeling/train_attrition_models.py`: huấn luyện, skalierung/encoder trong pipeline, so sánh mô hình, threshold tuning, lưu dự báo.

Lưu ý quan trọng: dữ liệu không được fit scaler/encoder trên toàn bộ tập dữ liệu trước khi train/test split. Mọi quy trình chuẩn hóa phải thực hiện trong pipeline trên tập train và áp dụng lên tập test.

## 10. Nhóm thực hiện

- Nhóm 7
- Môn: Khoa học dữ liệu
- Repo: `NewProGuest/KHDL-Group-7`
- Tác giả/owner repo: NewProGuest

## 11. Lưu ý phát triển

- Dữ liệu gốc không được chỉnh sửa trực tiếp.
- Mỗi script nên được chạy từ thư mục gốc repo.
- Khi thêm mô hình mới, lưu metrics và file dự đoán vào `data/reports/`.
- Nếu cần cập nhật mô hình trong `models/`, nên lưu cả metadata mô hình và file rõ ràng.

## 12. Tài liệu tham khảo

- IBM HR Analytics Employee Attrition & Performance (Kaggle)
- CRISP-DM
- Data Science lifecycle

---

Mục tiêu của dự án không chỉ là tạo ra một mô hình dự đoán mà còn là giúp phòng HR hiểu nguyên nhân và có kế hoạch giữ chân nhân sự một cách chủ động và hiệu quả.
