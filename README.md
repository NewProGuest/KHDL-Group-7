# KHDL-Group-7
Đồ án Khoa học dữ liệu: Phân tích nguyên nhân nghỉ việc của nhân sự (IBM HR Analytics Employee Attrition)

## Đề tài 5: Phân tích nguyên nhân nghỉ việc của nhân sự (HR Analytics)

### 📌 Tổng quan dự án
- Nhóm thực hiện: Nhóm 7 (KHDL)
- Bộ dữ liệu: [IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Mục tiêu chính:
  1. Khám phá các yếu tố ảnh hưởng đến quyết định nghỉ việc của nhân viên.
  2. Xây dựng mô hình dự đoán nguy cơ nghỉ việc của nhân sự.
  3. Đề xuất các giải pháp và chính sách giúp doanh nghiệp giữ chân nhân tài hiệu quả hơn.

### 🎯 Mục tiêu nghiên cứu
- Phân tích mối quan hệ giữa biến mục tiêu `Attrition` (Yes/No) và các biến độc lập như:
  - `MonthlyIncome`
  - `DistanceFromHome`
  - `JobSatisfaction`
  - `TotalWorkingYears`
  - `YearsAtCompany`
  - `WorkLifeBalance`
  - `EnvironmentSatisfaction`
  - `JobLevel`, `Department`, `Education`, ...
- Tìm ra các yếu tố có tác động lớn nhất tới khả năng nhân viên nghỉ việc.
- So sánh hiệu suất của các mô hình phân loại để chọn mô hình tối ưu.
- Đề xuất khuyến nghị phù hợp với doanh nghiệp dựa trên các dấu hiệu quan trọng từ dữ liệu.

### 🧪 Quy trình thực hiện

#### 1. Khám phá dữ liệu (EDA)
- Kiểm tra cấu trúc dữ liệu, kiểu dữ liệu và dữ liệu thiếu.
- Phân tích thống kê mô tả (mean, median, std, min, max).
- So sánh phân phối giữa nhóm nghỉ việc và không nghỉ việc.
- Trực quan hóa dữ liệu bằng:
  - Boxplot
  - Bar chart
  - Histogram
  - Heatmap
  - Scatter plot
- Xác định các biến có sự chênh lệch rõ ràng giữa hai nhóm.

#### 2. Tiền xử lý dữ liệu và Feature Engineering
- Xử lý dữ liệu thiếu nếu có.
- Chuyển đổi biến phân loại thành dạng phù hợp cho mô hình (One-Hot Encoding, Label Encoding).
- Tạo các biến mới để làm rõ hơn yếu tố ảnh hưởng, ví dụ:
  - Tỷ lệ tăng lương
  - Số năm làm việc tại công ty
  - Tỷ lệ thời gian ở công ty hiện tại so với tổng số năm kinh nghiệm
  - Tích hợp các biến tương tác quan trọng
- Cân nhắc kỹ thuật xử lý dữ liệu mất cân bằng lớp (`Class Imbalance`).

#### 3. Xây dựng mô hình dự đoán
- Áp dụng các mô hình phân loại phổ biến như:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - Gradient Boosting
- Tối ưu tham số bằng cách sử dụng:
  - Cross-validation
  - Grid Search / Random Search
- Đánh giá mô hình theo các tiêu chí:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - Confusion Matrix

#### 4. Phân tích độ quan trọng của đặc trưng
- Xác định các biến có ảnh hưởng lớn nhất đến việc nghỉ việc.
- Ví dụ: mức độ hài lòng công việc, khoảng cách từ nhà đến nơi làm việc, lương, số năm công tác, mức độ công nhận, cân bằng giữa công việc và cuộc sống.
- Dùng kết quả từ Feature Importance để giải thích nguyên nhân nghề nghiệp.

#### 5. Kết luận và khuyến nghị
- Tổng kết các yếu tố chính dẫn đến tình trạng nghỉ việc.
- Đề xuất các chính sách phù hợp để giảm tỷ lệ nghỉ việc, ví dụ:
  - Cải thiện chế độ đãi ngộ
  - Nâng cao môi trường làm việc
  - Tăng cơ hội thăng tiến
  - Quản lý cân bằng giữa công việc và cuộc sống
  - Tăng cường đánh giá sự hài lòng của nhân viên

### 📊 Kết quả mong đợi
- Hiểu rõ nguyên nhân chủ yếu khiến nhân viên nghỉ việc.
- Xây dựng mô hình dự đoán nguy cơ nghỉ việc với độ chính xác cao.
- Đưa ra các đề xuất thực tiễn giúp doanh nghiệp cải thiện tỷ lệ giữ chân nhân tài.

### 🛠️ Công cụ và thư viện dự kiến
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

### 📁 Cấu trúc dự án
```bash
KHDL-Group-7/
├── data/
│   └── HR_Employee_Attrition_Data.csv
├── notebooks/
│   └── EDA_and_Modeling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── modeling.py
│   └── visualization.py
├── README.md
└── requirements.txt
```

### 🔗 Tài liệu tham khảo
- [IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
