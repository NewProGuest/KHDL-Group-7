# 📊 IBM HR Analytics: Employee Attrition Prediction

> Đồ án Khoa học Dữ liệu (Data Science Final Project - INFO3020)  
> Phân tích nguyên nhân, đánh giá chất lượng dữ liệu và xây dựng mô hình dự báo nhân sự nghỉ việc tại tập đoàn IBM — Nhóm 7.

---

## 1. Giới thiệu dự án

Dự án này được triển khai theo đúng quy trình của vòng đời khoa học dữ liệu (Data Science Lifecycle), bắt đầu từ việc xác định bài toán kinh doanh, thu thập và làm sạch dữ liệu, khám phá dữ liệu, xây dựng mô hình học máy, đánh giá hiệu quả và đề xuất giải pháp hỗ trợ ra quyết định cho phòng nhân sự.

Mục tiêu chính của dự án là xây dựng một hệ thống phân tích và dự báo khả năng nhân sự nghỉ việc (Attrition) dựa trên các thuộc tính về nhân khẩu học, thâm niên, lương, đánh giá hiệu suất, mức độ hài lòng công việc, môi trường làm việc và các yếu tố khác.

---

## 2. Business Understanding (Hiểu bài toán kinh doanh)

### 2.1. Bài toán kinh doanh
Bài toán đặt ra cho doanh nghiệp là:

> Xây dựng mô hình phân tích và dự đoán khả năng nhân viên nghỉ việc (`Attrition = Yes/No`) dựa trên các thuộc tính nhân sự, hiệu suất và cảm nhận về công việc.

### 2.2. Lý do cần giải quyết
Tỷ lệ nhân sự nghỉ việc cao tạo ra nhiều hệ quả tiêu cực cho doanh nghiệp, bao gồm:

- Chi phí tuyển dụng và đào tạo nhân sự mới
- Mất năng suất và hiệu quả vận hành
- Giảm sự ổn định của đội ngũ
- Tăng gánh nặng cho quản lý và phòng nhân sự
- Làm suy giảm năng lực cạnh tranh và chất lượng dịch vụ

### 2.3. Mục tiêu của dự án
- Phân tích nguyên nhân dẫn đến nghỉ việc của nhân viên
- Xác định các nhóm nhân viên có nguy cơ nghỉ việc cao
- Dự đoán sớm các trường hợp có nguy cơ rời bỏ công ty
- Tạo cơ sở dữ liệu định lượng hỗ trợ cho các quyết định của HR và ban lãnh đạo
- Đề xuất các chính sách giữ chân nhân tài hiệu quả

### 2.4. Đối tượng liên quan (Stakeholders)
- Phòng Nhân sự (HR)
- Ban quản lý cấp cao
- Quản lý trực tiếp / trưởng bộ phận
- Các nhà khoa học dữ liệu và kỹ sư dữ liệu trong dự án

### 2.5. Kết quả mong muốn
- Một mô hình dự báo nhân sự nghỉ việc có độ tin cậy cao
- Bảng phân tích yếu tố ảnh hưởng quan trọng
- Danh sách nhân viên có nguy cơ nghỉ việc cao
- Gợi ý hành động thực tiễn cho doanh nghiệp

### 2.6. KPI thành công
- Độ chính xác của mô hình: Accuracy
- Độ nhạy và độ đặc hiệu: Precision, Recall, F1-score
- Độ phân biệt của mô hình: AUC-ROC
- Khả năng phát hiện đúng các trường hợp attrition cao
- Tỷ lệ giảm attrition trong quá trình áp dụng khuyến nghị thực tế

---

## 3. Thu thập dữ liệu nguồn

### 3.1. Nguồn dữ liệu
Bộ dữ liệu sử dụng trong dự án là:

- IBM HR Analytics Employee Attrition & Performance
- Nguồn: Kaggle
- Dữ liệu gốc được tải về và lưu trữ trong dự án để phục vụ phân tích, tiền xử lý và mô hình hóa

### 3.2. Thông tin bộ dữ liệu
- Số dòng: 1.470
- Số cột: 35
- Biến mục tiêu: `Attrition` (Yes / No)

### 3.3. Các nhóm thuộc tính chính
- Nhân khẩu học & cá nhân: `Age`, `Gender`, `Education`, `MaritalStatus`, `DistanceFromHome`
- Công việc & thâm niên: `Department`, `JobRole`, `JobLevel`, `YearsAtCompany`, `TotalWorkingYears`
- Thu nhập & phúc lợi: `MonthlyIncome`, `MonthlyRate`, `StockOptionLevel`, `PercentSalaryHike`
- Đánh giá & mức độ hài lòng: `JobSatisfaction`, `EnvironmentSatisfaction`, `WorkLifeBalance`, `PerformanceRating`
- Hiệu suất & phát triển: `YearsSinceLastPromotion`, `TrainingTimesLastYear`, `JobInvolvement`

### 3.4. Quy trình thu thập
1. Tải dữ liệu gốc từ nguồn chính thức (Kaggle)
2. Lưu dữ liệu vào thư mục `data/raw/`
3. Ghi thông tin nguồn dữ liệu, thời gian tải và mô tả bộ dữ liệu
4. Kiểm tra cấu trúc, kiểu dữ liệu và giá trị khuyết
5. Tạo dữ liệu đã xử lý cho các bước EDA và modeling

---

## 4. Kiến trúc lưu trữ dữ liệu

Để đảm bảo quy trình làm việc rõ ràng và dễ quản lý, dự án nên tổ chức dữ liệu theo cấu trúc sau:

```text
KHDL-Group-7/
├── data/
│   ├── raw/
│   │   ├── HR_Employee_Attrition_Data.csv
│   │   ├── data_dictionary.csv
│   │   └── metadata.json
│   ├── processed/
│   │   ├── cleaned_data.csv
│   │   ├── train.csv
│   │   └── test.csv
│   └── reports/
│       ├── eda_summary.html
│       ├── missing_values_report.csv
│       └── feature_importance_report.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── utils.py
├── docs/
│   ├── business_understanding.md
│   ├── data_dictionary.md
│   └── project_plan.md
├── config/
│   ├── model_config.yaml
│   └── paths.yaml
├── .github/
│   └── workflows/
│       └── ci.yml
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── .dvc/   # nếu sử dụng DVC quản lý dữ liệu
```

### 4.1. Nguyên tắc lưu trữ
- Dữ liệu thô (`raw`) không được chỉnh sửa trực tiếp
- Dữ liệu đã xử lý được lưu riêng (`processed`)
- Mỗi phiên bản dữ liệu nên có ghi chú rõ ràng về ngày và nội dung thay đổi
- Cần lưu metadata để trace nguồn gốc dữ liệu

### 4.2. Đảm bảo tính toàn vẹn dữ liệu
- Kiểm tra checksum hoặc hash cho file dữ liệu quan trọng
- Lưu trữ dữ liệu theo từng version
- Theo dõi lịch sử thay đổi bằng Git hoặc DVC

---

## 5. Quản lý phiên bản mã nguồn trên GitHub

### 5.1. Mô hình nhánh
Để làm việc nhóm hiệu quả, nên áp dụng mô hình nhánh chuẩn:

- `main`: nhánh chính, chứa phiên bản ổn định
- `dev`: nhánh tích hợp nội bộ
- `feature/...`: nhánh cho từng tính năng hoặc module
- `hotfix/...`: nhanh chóng sửa lỗi trong production

### 5.2. Quy tắc commit
Các commit nên theo format rõ ràng, ví dụ:

- `feat: add data preprocessing pipeline`
- `fix: handle missing values in monthly income`
- `docs: update business understanding section`
- `data: add raw HR attrition dataset snapshot`
- `test: add model evaluation metrics notebook`

### 5.3. Quy trình làm việc nhóm
1. Tạo nhánh mới cho từng nhiệm vụ
2. Thực hiện phát triển trên nhánh riêng
3. Tạo Pull Request (PR)
4. Review code và đảm bảo không có lỗi logic
5. Merge vào `dev` sau khi đã được confirm
6. Merge `dev` vào `main` khi hoàn thành milestone

### 5.4. Quản lý dữ liệu lớn bằng Git
Nếu dataset trở nên lớn, nên sử dụng:

- Git LFS để quản lý file có kích thước lớn
- DVC để theo dõi version dữ liệu và mô hình

Điều này giúp repository không quá nặng và vẫn quản lý được dữ liệu khoa học dữ liệu chuyên nghiệp.

---

## 6. Quy trình phân tích dữ liệu dự kiến

### Giai đoạn 1: Nghiên cứu bài toán
- Xác định business problem
- Làm rõ mục tiêu và KPI
- Định nghĩa target variable

### Giai đoạn 2: Thu thập dữ liệu
- Tải dataset từ Kaggle
- Ghi lại source và metadata
- Kiểm tra độ đầy đủ dữ liệu

### Giai đoạn 3: Khám phá dữ liệu (EDA)
- Phân tích thống kê mô tả
- Quan sát phân phối từng biến
- Xác định mối tương quan với Attrition
- Phát hiện outlier và missing values

### Giai đoạn 4: Tiền xử lý dữ liệu
- Xử lý dữ liệu thiếu
- Chuyển dữ liệu categorical thành dạng số
- Chuẩn hóa / scaling nếu cần
- Chia train/test set

### Giai đoạn 5: Xây dựng mô hình
- So sánh các mô hình như: Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, v.v.
- Đánh giá theo các métrics phù hợp

### Giai đoạn 6: Đánh giá và giải thích
- So sánh hiệu suất mô hình
- Phân tích feature importance
- Trình bày giải thích dựa trên yếu tố ảnh hưởng lớn nhất đến nghỉ việc

### Giai đoạn 7: Đề xuất chiến lược
- HR nên can thiệp ở nhóm nào?
- Các yếu tố nào là nguy cơ chính?
- Cách đưa ra quyết định dựa trên mô hình và dữ liệu

---

## 7. Công cụ và kỹ thuật dự kiến

- Ngôn ngữ lập trình: Python
- Thư viện: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost
- Notebook: Jupyter / Google Colab
- Quản lý mã nguồn: GitHub
- Tự động hóa: GitHub Actions
- Quản lý dữ liệu: DVC hoặc Git LFS

---

## 8. Kết luận

Dự án IBM HR Analytics Employee Attrition Prediction là một bài toán kinh điển trong khoa học dữ liệu với mục tiêu thực tiễn rất rõ ràng: dự báo và giải thích nguyên nhân nhân sự nghỉ việc. Dự án không chỉ tập trung vào kỹ thuật mô hình mà còn cần đặt trọng tâm vào việc hiểu kinh doanh, kiểm soát chất lượng dữ liệu và xây dựng quy trình làm việc khoa học, chuyên nghiệp.

Việc tổ chức dữ liệu khoa học, quản lý phiên bản mã nguồn trên GitHub và chia nhánh phát triển rõ ràng sẽ giúp nhóm thực hiện dự án hiệu quả, dễ kiểm soát và dễ mở rộng trong tương lai.

---

## 9. Tài liệu tham khảo

- Kaggle: IBM HR Analytics Employee Attrition & Performance
- CRISP-DM methodology
- Data Science Lifecycle
- GitHub best practices for collaborative projects

---

## 10. Nhóm thực hiện

- Nhóm 7
- Môn: Khoa học dữ liệu
- Dự án: Phân tích nguyên nhân nghỉ việc của nhân sự

