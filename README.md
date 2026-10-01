# 📊 Đồ án Khoa học Dữ liệu: Phân tích nguyên nhân nghỉ việc của nhân sự (HR Analytics)

> **Nhóm thực hiện:** Nhóm 7 (KHDL)  
> **Repository:** [KHDL-Group-7](https://github.com/NewProGuest/KHDL-Group-7)

---

## 🎯 1. Mục tiêu nghiên cứu
- **Phân tích nguyên nhân:** Tìm hiểu các yếu tố ảnh hưởng đến quyết định nghỉ việc của nhân viên (`Attrition`).
- **Mô hình hóa dự đoán:** Xây dựng mô hình phân loại để cảnh báo sớm nguy cơ nghỉ việc.
- **Đề xuất giải pháp:** Đưa ra khuyến nghị cho phòng HR nhằm giữ chân nhân sự.

---

## 📦 2. Bộ dữ liệu
- **Tên dataset:** [IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (dữ liệu giả lập do IBM tạo ra)
- **Kích thước:** 1,470 bản ghi, 35 thuộc tính. Sau khi bỏ 4 cột không mang thông tin (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`) và thêm 6 biến phái sinh, mô hình dùng 36 đặc trưng.
- **Biến mục tiêu:** `Attrition` (`Yes`: 16.1%, `No`: 83.9% – bài toán mất cân bằng lớp).

---

## 📈 3. Kết quả phân tích chính

> Toàn bộ số liệu dưới đây được sinh từ `notebooks/01_HR_Analytics.ipynb` (`random_state=42`).

### 3.1. So sánh nhóm nghỉ việc và ở lại (EDA)
Tỷ lệ nghỉ việc trung bình là **16.1%**.
1. **Làm thêm giờ (`OverTime`):** 30.5% (có OT) so với 10.4% (không OT), gấp khoảng **3 lần**.
2. **Thu nhập (`MonthlyIncome`):** dưới $3,000/tháng nghỉ việc 28.6%, từ $3,000 trở lên 11.5%; nhóm lương thấp chiếm 47.7% số người nghỉ việc. Trung vị thu nhập: 3,202 (nghỉ việc) so với 5,204 (ở lại).
3. **Mức hài lòng:** `JobSatisfaction` = 1 nghỉ việc 22.8% (mức 4: 11.3%); `EnvironmentSatisfaction` = 1: 25.4%; `WorkLifeBalance` = 1: 31.2%.
4. **Công tác (`BusinessTravel`):** `Travel_Frequently` 24.9% so với `Non-Travel` 8.0%.
5. **Quyền chọn cổ phiếu (`StockOptionLevel`):** mức 0 nghỉ việc 24.4%, mức 1–2 chỉ 7.6–9.4%.
6. **Khoảng cách đi làm (`DistanceFromHome`):** sống xa hơn 10 km nghỉ việc 20.9% so với 14.0%. Tác động vừa phải.
7. **Phòng ban (`Department`):** Sales 20.6% (n=446), Human Resources 19.0% (n=63), Research & Development 13.8% (n=961). Lưu ý nhóm HR chỉ có 63 người nên tỷ lệ kém ổn định.

### 3.2. Hiệu suất mô hình Machine Learning
Quy trình: Feature Engineering (6 biến mới) → chia train/test 80/20 có phân tầng → so sánh 3 mô hình bằng Stratified 5-fold CV → tinh chỉnh bằng `RandomizedSearchCV` → chọn ngưỡng quyết định.

| Mô hình | CV ROC-AUC (trước tinh chỉnh) | CV ROC-AUC (sau tinh chỉnh) |
|---|---|---|
| Logistic Regression | 0.829 | 0.830 |
| Random Forest | 0.810 | 0.813 |
| XGBoost | 0.812 | **0.831** |

- **Mô hình được chọn:** XGBoost. Lưu ý Logistic Regression gần như ngang bằng, nên mô hình phức tạp không mang lại lợi thế rõ rệt.
- **ROC-AUC:** 0.831 (CV trên tập train) | **0.779** (tập test 20% giữ riêng, 47 mẫu dương).
- **PR-AUC (test):** 0.476, so với mức cơ sở 0.160 khi đoán ngẫu nhiên.
- **Ngưỡng quyết định:** 0.30, chọn để Recall ≥ 70% trên out-of-fold. Trên tập test: Recall = 0.64, Precision = 0.38.
- *Accuracy không được dùng làm chỉ số chính, vì đoán "No" cho tất cả cũng đã đạt ~84%.*

### 3.3. Độ quan trọng của các đặc trưng
**Kết quả chính – Permutation Importance trên tập test** (20 lần lặp, mức giảm ROC-AUC khi xáo trộn đặc trưng, mô hình XGBoost):

| Hạng | Đặc trưng | Giảm AUC |
|---|---|---|
| 1 | `OverTime` | 0.106 ± 0.017 |
| 2 | `StockOptionLevel` | 0.029 ± 0.008 |
| 3 | `MonthlyIncome` | 0.016 ± 0.011 |
| 4 | `AvgSatisfaction` (biến phái sinh) | 0.016 ± 0.010 |
| 5 | `BusinessTravel` | 0.013 ± 0.006 |

`OverTime` vượt trội rõ rệt so với các biến còn lại; từ hạng 3 trở đi chênh lệch nhỏ hơn độ lệch chuẩn nên không nên diễn giải thứ hạng quá chi tiết.

**Mô hình minh họa – Decision Tree (max_depth=5, `class_weight="balanced"`, huấn luyện trên toàn bộ 1,470 bản ghi; Mục 9 của notebook):**

| Hạng | Đặc trưng | Độ quan trọng |
|---|---|---|
| 1 | `OverTime_Yes` | 21.71% |
| 2 | `JobHopper` (biến phái sinh: số công ty đã làm / số năm làm việc) | 14.48% |
| 3 | `JobLevel` | 9.51% |
| 4 | `JobRole_Sales Executive` | 6.43% |
| 5 | `TotalWorkingYears` | 5.73% |

Đây là độ quan trọng *in-sample* dựa trên impurity của một cây đơn, thiên về các biến liên tục và chưa được kiểm chứng trên dữ liệu test, nên chỉ mang tính minh họa. Chỉ `OverTime` nhất quán ở cả hai phương pháp. `JobHopper` đứng thứ 2 trong cây nhưng gần như không đóng góp (hạng 28/36, mức giảm AUC ≈ −0.002) trong permutation importance, tức là khả năng cao chỉ là hiện tượng học thuộc của cây. Tương tự, `DailyRate` chiếm 3.44% trong cây nhưng xếp hạng **34/36** (đóng góp âm) trong permutation importance, nên không được đưa vào khuyến nghị.

*Lưu ý:* độ quan trọng phản ánh đóng góp vào dự đoán, không chứng minh quan hệ nhân quả. Các biến tương quan mạnh (`Age`, `TotalWorkingYears`, `MonthlyIncome`, `JobLevel`) có thể chia sẻ mức quan trọng với nhau.

### 3.4. Danh sách nhân sự nguy cơ cao
Với ngưỡng 0.30, mô hình gắn cờ 430/1,470 nhân viên (điểm out-of-fold). Nhóm bị gắn cờ có tỷ lệ làm thêm giờ **52%** (so với 19%), thu nhập trung bình $4,518 (so với $7,324). Bảng xếp hạng chi tiết theo `EmployeeNumber` nằm ở Mục 10 của notebook.

---

## 💡 4. Khuyến nghị cho Doanh nghiệp (HR Policies)
- ⚖️ **Cân bằng tải công việc:** giới hạn làm thêm giờ, ưu tiên nhóm bị gắn cờ. Đây là yếu tố có bằng chứng mạnh nhất.
- 💵 **Rà soát đãi ngộ:** nâng lương sàn cho vị trí thu nhập dưới $3,000 và mở rộng stock option cho nhân viên cấp thấp.
- 🤝 **Theo dõi mức hài lòng:** họp 1-1 với nhân viên có điểm hài lòng 1–2, đặc biệt nhóm vừa làm thêm giờ vừa hài lòng thấp (nghỉ việc 36.6%).
- ✈️ **Giảm áp lực công tác:** luân phiên nhân sự đi công tác thường xuyên.
- 🚗 **Hỗ trợ đi lại (ưu tiên thấp hơn):** phụ cấp hoặc làm việc linh hoạt cho nhân sự sống xa.

## ⚠️ 5. Hạn chế
- Dữ liệu IBM là dữ liệu giả lập; các mối quan hệ là tương quan, cần thử nghiệm chính sách thực tế trước khi kết luận nhân quả.
- ROC-AUC trên test (0.779) thấp hơn CV (0.831) và tập test chỉ có 47 mẫu dương, nên con số có độ dao động đáng kể.
- Với Recall ≥ 70%, Precision chỉ khoảng 0.43 trên out-of-fold (0.38 trên tập test): hơn một nửa số người bị gắn cờ sẽ không nghỉ việc, cần cân nhắc chi phí can thiệp.

## ▶️ 6. Cách chạy lại
```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_HR_Analytics.ipynb
```
File dữ liệu phải nằm ở `data/WA_Fn-UseC_-HR-Employee-Attrition.csv` (đủ 1,470 dòng).
