# IBM HR Analytics: Employee Attrition Prediction

> Đồ án Khoa học Dữ liệu (Data Science Final Project - INFO3020)  
> Phân tích nguyên nhân, đánh giá chất lượng dữ liệu và xây dựng mô hình dự báo nhân sự nghỉ việc tại tập đoàn IBM — Nhóm 7.

---

## 1. Giới thiệu dự án

Dự án này tập trung vào việc phân tích dữ liệu nhân sự và xây dựng mô hình dự đoán khả năng nhân viên nghỉ việc (Attrition). Mục tiêu là giúp phòng nhân sự và ban quản lý nhận diện sớm các trường hợp có nguy cơ nghỉ việc, hiểu nguyên nhân chủ yếu, và đề xuất các biện pháp phù hợp để giữ chân nhân tài.

---

## 2. Business Understanding

### 2.1. Bài toán kinh doanh

> Xây dựng mô hình dự đoán khả năng nhân viên nghỉ việc (`Attrition = Yes/No`) dựa trên các thuộc tính về nhân sự, hiệu suất, cảm nhận, và yếu tố môi trường làm việc.

### 2.2. Lý do cần giải quyết

Tỷ lệ nghỉ việc cao gây ra nhiều tác động tiêu cực cho doanh nghiệp, bao gồm:

- Chi phí tuyển dụng và đào tạo nhân sự mới
- Mất năng suất và kiến thức chuyên môn
- Giảm độ ổn định của đội ngũ
- Tăng gánh nặng cho quản lý và phòng nhân sự
- Ảnh hưởng đến hiệu quả hoạt động và cạnh tranh

### 2.3. Mục tiêu của dự án

- Phân tích các yếu tố ảnh hưởng đến việc nhân viên nghỉ việc
- Xác định nhóm nhân viên có nguy cơ nghỉ việc cao
- Dự đoán sớm các trường hợp có nguy cơ rời công ty
- Hỗ trợ HR và ban lãnh đạo đưa ra quyết định kịp thời
- Đề xuất các chính sách giữ chân nhân tài hiệu quả

### 2.4. Stakeholders

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

- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC
- Khả năng phát hiện đúng các trường hợp nghỉ việc có nguy cơ cao

---

## 3. Thu thập dữ liệu nguồn

### 3.1. Nguồn dữ liệu

Bộ dữ liệu chính được sử dụng trong dự án là:

- IBM HR Analytics Employee Attrition & Performance
- Nguồn: Kaggle

### 3.2. Thông tin bộ dữ liệu

- Số dòng: 1.470
- Số cột: 35
- Biến mục tiêu: `Attrition` (Yes / No)

### 3.3. Các nhóm thuộc tính chính

- Nhân khẩu học: `Age`, `Gender`, `Education`, `MaritalStatus`, `DistanceFromHome`
- Công việc và thâm niên: `Department`, `JobRole`, `JobLevel`, `YearsAtCompany`, `TotalWorkingYears`
- Thu nhập và phúc lợi: `MonthlyIncome`, `MonthlyRate`, `StockOptionLevel`, `PercentSalaryHike`
- Đánh giá và mức độ hài lòng: `JobSatisfaction`, `EnvironmentSatisfaction`, `WorkLifeBalance`, `PerformanceRating`
- Hiệu suất và phát triển: `YearsSinceLastPromotion`, `TrainingTimesLastYear`, `JobInvolvement`

### 3.4. Quy trình thu thập

1. Tải dữ liệu gốc từ Kaggle
2. Lưu dữ liệu vào thư mục `data/raw/`
3. Ghi rõ nguồn, ngày tải và mô tả bộ dữ liệu
4. Kiểm tra cấu trúc dữ liệu, kiểu dữ liệu và giá trị thiếu
5. Tạo tập dữ liệu đã xử lý để phục vụ EDA và mô hình hóa

---

## 4. Cấu trúc kho dữ liệu tối giản

Toàn bộ dự án được tổ chức theo cấu trúc đơn giản sau:

```text
KHDL-Group-7/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
├── notebooks/
├── src/
├── docs/
└── models/
```

### Mục đích của từng phần

- `data/raw/`: dữ liệu thô gốc, không chỉnh sửa
- `data/processed/`: dữ liệu đã xử lý, sạch, sẵn sàng cho phân tích
- `data/reports/`: báo cáo EDA, báo cáo missing value, báo cáo feature importance
- `notebooks/`: phân tích và mô hình hóa bằng Jupyter Notebook
- `src/`: code Python cho preprocessing, modeling, evaluation
- `docs/`: tài liệu về bài toán, dictionary dữ liệu, báo cáo tiến độ
- `models/`: lưu mô hình đã huấn luyện

---

## 5. Quy trình phân tích dữ liệu dự kiến

### Giai đoạn 1: Nghiên cứu bài toán
- Xác định business problem
- Làm rõ mục tiêu và KPI
- Định nghĩa biến mục tiêu

### Giai đoạn 2: Thu thập dữ liệu
- Tải dữ liệu từ Kaggle
- Ghi metadata và nguồn gốc
- Kiểm tra độ đầy đủ dữ liệu

### Giai đoạn 3: Khám phá dữ liệu (EDA)
- Phân tích thống kê mô tả
- Quan sát phân phối từng biến
- Xác định mối tương quan với Attrition
- Phát hiện missing values và outlier

### Giai đoạn 4: Tiền xử lý dữ liệu
- Xử lý dữ liệu thiếu
- Chuyển dữ liệu categorical về dạng số
- Chia tập train/test

### Giai đoạn 5: Xây dựng mô hình
- So sánh các mô hình như Logistic Regression, Decision Tree, Random Forest, XGBoost
- Đánh giá theo các métrics phù hợp

### Giai đoạn 6: Đánh giá và giải thích
- So sánh hiệu suất mô hình
- Phân tích feature importance
- Trình bày các yếu tố ảnh hưởng lớn nhất đến nghỉ việc

### Giai đoạn 7: Đề xuất chiến lược
- Xác định nhóm nhân viên có nguy cơ cao
- Gợi ý các hành động can thiệp sớm của HR

---

## 6. Tiền xử lý dữ liệu

Hai bước tiền xử lý được tách thành các script và file đầu ra riêng:

1. `python src/preprocessing/handle_outliers.py` đọc dữ liệu gốc, chặn ngoại lai theo ngưỡng IQR (`Q1 - 1.5 × IQR`, `Q3 + 1.5 × IQR`) cho các biến liên tục. Ngưỡng được làm tròn vào trong cho phù hợp với các cột số nguyên, và script tạo:
   - `data/processed/hr_attrition_outliers.csv`
   - `data/reports/outlier_treatment_report.csv` (ngưỡng và số giá trị bị chặn theo biến)
2. `python src/preprocessing/transform_data.py` đọc file đã xử lý ngoại lai, chuẩn hóa các biến liên tục bằng `StandardScaler`, one-hot encode biến phân loại, mã hóa `Attrition` (`No = 0`, `Yes = 1`), và tạo `data/processed/hr_attrition_transformed.csv`.

Các cột hằng `EmployeeCount`, `Over18`, `StandardHours` được loại khỏi dữ liệu biến đổi; `EmployeeNumber` được giữ làm định danh và không chuẩn hóa. Các biến thứ bậc dạng số được giữ nguyên mã thứ bậc.

Các scaler và bộ mã hóa trong script được fit trên toàn bộ dữ liệu để tạo file tiền xử lý dùng cho phân tích. Khi đánh giá mô hình, hãy chia train/test trước và chỉ fit các phép biến đổi trên tập train để tránh rò rỉ dữ liệu.

## 7. Công cụ và kỹ thuật dự kiến

- Ngôn ngữ lập trình: Python
- Thư viện: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, missingno
- Notebook: Jupyter / Google Colab
- Quản lý mã nguồn: GitHub

---

## 8. Tài liệu tham khảo

- Kaggle: IBM HR Analytics Employee Attrition & Performance
- CRISP-DM methodology
- Data Science Lifecycle

---

## 9. Nhóm thực hiện

- Nhóm 7
- Môn: Khoa học dữ liệu
- Dự án: Phân tích nguyên nhân nghỉ việc của nhân sự

---

## 10. Lưu ý triển khai nhóm

- Mỗi nhiệm vụ nên được phát triển trên nhánh riêng
- Commit nên mô tả rõ mục đích công việc
- PR cần được review trước khi merge
- Dữ liệu gốc không được chỉnh sửa trực tiếp
- Mỗi milestone nên có một báo cáo tóm tắt rõ ràng

Mục tiêu của dự án không chỉ là xây dựng mô hình dự đoán, mà còn là hiểu được yếu tố nào đang khiến nhân viên nghỉ việc và giúp doanh nghiệp có thể phản ứng kịp thời bằng các quyết định thực tế.
