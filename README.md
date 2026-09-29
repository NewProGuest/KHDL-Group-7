# 📊 Đồ án Khoa học Dữ liệu: Phân tích nguyên nhân nghỉ việc của nhân sự (HR Analytics)

> **Nhóm thực hiện:** Nhóm 7 (KHDL)  
> **Repository:** [KHDL-Group-7](https://github.com/NewProGuest/KHDL-Group-7)

---

## 🎯 1. Mục tiêu nghiên cứu
- **Phân tích nguyên nhân:** Tìm hiểu các yếu tố cốt lõi ảnh hưởng đến quyết định nghỉ việc của nhân viên (`Attrition`).
- **Mô hình hóa dự đoán:** Xây dựng mô hình Machine Learning phân loại để cảnh báo sớm nguy cơ nghỉ việc.
- **Đề xuất giải pháp:** Đưa ra khuyến nghị cho phòng HR nhằm giảm tỷ lệ nghỉ việc và giữ chân nhân tài.

---

## 📦 2. Bộ dữ liệu
- **Tên dataset:** [IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Kích thước:** 1,470 bản ghi, 35 thuộc tính (gồm thông tin lương, khoảng cách đi làm, chỉ số hài lòng, số năm kinh nghiệm,...).
- **Biến mục tiêu:** `Attrition` (`Yes`: 16%, `No`: 84% - Bài toán bị mất cân bằng lớp).

---

## 📈 3. Kết quả phân tích chính (Key Insights & Findings)

### 3.1. Các yếu tố tác động mạnh nhất đến việc nghỉ việc (EDA)
1. **Tình trạng làm thêm giờ (`OverTime`):** Nhân viên làm thêm giờ có tỷ lệ nghỉ việc cao gấp **3 lần** so với nhóm không làm thêm giờ.
2. **Mức thu nhập hàng tháng (`MonthlyIncome`):** Nhóm nghỉ việc tập trung chủ yếu ở phân khúc lương thấp (dưới $3,000/tháng).
3. **Mức độ hài lòng công việc (`JobSatisfaction` & `EnvironmentSatisfaction`):** Điểm hài lòng ở mức 1 (Rất thấp) có nguy cơ nghỉ việc vượt trội.
4. **Khoảng cách đi làm (`DistanceFromHome`):** Nhân viên sống xa công ty (>10km) có xu hướng rời đi cao hơn.

### 3.2. Hiệu suất mô hình Machine Learning
- **Mô hình sử dụng:** Random Forest Classifier (kèm xử lý `class_weight='balanced'`).
- **Đánh giá:**
  - **Accuracy:** ~85%
  - **ROC-AUC Score:** ~0.81
  - **Top 3 đặc trưng quan trọng nhất:** `OverTime`, `MonthlyIncome`, `TotalWorkingYears`.

---

## 💡 4. Khuyến nghị cho Doanh nghiệp (HR Policies)
- ⚖️ **Cân bằng tải công việc:** Tối ưu hóa quy trình làm việc để giảm tải tình trạng làm thêm giờ (`OverTime`) kéo dài.
- 💵 **Rà soát chính sách đãi ngộ:** Điều chỉnh mức lương sàn cho các vị trí có thu nhập thấp và thâm niên cao.
- 🚗 **Hỗ trợ di chuyển:** Cung cấp phụ cấp xe đưa đón hoặc chế độ làm việc linh hoạt (Hybrid/Remote) cho nhân sự ở xa.