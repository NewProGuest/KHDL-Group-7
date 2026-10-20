# 📊 Đồ án Khoa học Dữ liệu: Phân tích nguyên nhân nghỉ việc của nhân sự (HR Analytics)

> Nhóm thực hiện: Nhóm 7 (KHDL)
> Repository: https://github.com/NewProGuest/KHDL-Group-7

---

## 1. Mục tiêu nghiên cứu
- Phân tích các yếu tố ảnh hưởng đến quyết định nghỉ việc của nhân viên (`Attrition`).
- Xây dựng mô hình phân loại để dự đoán nguy cơ nghỉ việc.
- Đề xuất giải pháp cho phòng HR nhằm giảm tỷ lệ nghỉ việc.

---

## 2. Bộ dữ liệu
- Tên dataset: IBM HR Analytics Employee Attrition
- Số dòng: 1,470
- Số cột: 35
- Biến mục tiêu: `Attrition` (`Yes` / `No`)
- Dữ liệu có tính mất cân bằng lớp: phần lớn nhân viên ở lại, chỉ khoảng 16% nghỉ việc.

### 2.1. Dữ liệu thiếu
Ta kiểm tra giá trị thiếu bằng `df.isnull().sum()` và `df.isnull().mean() * 100`.

Kết quả của phiên bản gốc của dataset IBM HR Analytics thường là:
- không có cột nào chứa `NaN`
- không cần imputation trong bước đầu tiên
- không cần xóa hàng hoặc xóa cột vì không có missing values

---

## 3. Quá trình EDA

### 3.1. So sánh nhóm nghỉ việc và ở lại
Các biến quan sát được có xu hướng cho thấy:
- `MonthlyIncome` của nhóm nghỉ việc thấp hơn rõ rệt so với nhóm ở lại
- `Age`, `TotalWorkingYears`, `YearsAtCompany` của nhóm nghỉ việc thấp hơn
- `OverTime` có mối liên hệ mạnh với khả năng nghỉ việc
- `JobSatisfaction` thấp và `BusinessTravel` thường xuyên cũng liên quan tới tỷ lệ nghỉ việc cao hơn

### 3.2. Dữ liệu thiếu / chất lượng dữ liệu
Trong phiên bản gốc dataset:
- không có missing values
- không có duplicate row đáng kể
- cần bỏ các cột vô nghĩa và cột ID trước khi huấn luyện mô hình

Các cột được loại bỏ gồm:
- `EmployeeCount`
- `Over18`
- `StandardHours`
- `EmployeeNumber`

---

## 4. Tiền xử lý và feature engineering
- Bỏ các cột hằng số và cột ID
- Tạo các biến mới để biểu diễn mức độ gắn bó và sự chuyển đổi công việc:
  - `IncomePerLevel`
  - `LoyaltyRatio`
  - `AvgSatisfaction`
  - `JobHopper`
- Chia dữ liệu thành train/test với `stratify=y` để giữ tỉ lệ lớp nhất quán
- Dùng `ColumnTransformer` để xử lý dữ liệu số và phân loại đồng thời

---

## 5. Mô hình hóa
Các mô hình được so sánh:
- Logistic Regression
- Random Forest
- XGBoost

### 5.1. Chỉ số đánh giá
Vì dữ liệu mất cân bằng, mô hình không nên đánh giá dựa trên Accuracy đơn thuần. Ta sử dụng:
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

### 5.2. Kết quả mẫu
Sau khi chạy notebook, mô hình cần báo ra kết quả tương tự dạng sau:

| Mô hình | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | ... | ... | ... | ... | ... | ... |
| Random Forest | ... | ... | ... | ... | ... | ... |
| XGBoost | ... | ... | ... | ... | ... | ... |

Mô hình tốt nhất sẽ được chọn theo `ROC-AUC` hoặc `PR-AUC` thay vì chỉ Accuracy.

---

## 6. Feature importance
Sử dụng permutation importance để xem biến nào tác động mạnh nhất tới khả năng nghỉ việc.

Các biến thường có xu hướng nổi bật bao gồm:
- `OverTime`
- `MonthlyIncome`
- `JobLevel`
- `Age`
- `WorkLifeBalance`
- `JobSatisfaction`
- `BusinessTravel`

---

## 7. Khuyến nghị cho HR
- Tăng cường kiểm soát làm thêm giờ (`OverTime`)
- Tái rà soát mức lương và cơ cấu đãi ngộ
- Theo dõi nhân viên có mức độ hài lòng thấp
- Hỗ trợ nhân viên đi công tác và cân bằng thời gian làm việc
- Theo dõi các nhân viên có nguy cơ cao theo xác suất dự đoán

---

## 8. Hạn chế
- Dữ liệu IBM là dữ liệu mô phỏng, không hoàn toàn phản ánh môi trường thực tế
- Dữ liệu mất cân bằng khiến các metric như Accuracy dễ gây hiểu nhầm
- Kết quả mô hình có thể thay đổi tùy hệ số và lựa chọn ngưỡng quyết định

---

## 9. Cách chạy lại
```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_HR_Analytics.ipynb
```

Đảm bảo dữ liệu nằm ở:
`data/WA_Fn-UseC_-HR-Employee-Attrition.csv`

---

## 10. Tóm tắt
Đây là bài toán phân loại nhị phân về nghỉ việc của nhân viên. Cách tiếp cận đúng là:
- audit dữ liệu
- kiểm tra missing values
- EDA để tìm yếu tố nguy cơ
- tiền xử lý
- huấn luyện và đánh giá mô hình theo metric phù hợp
- đưa ra khuyến nghị thực tế cho HR

