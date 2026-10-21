# 📊 IBM HR Analytics: Employee Attrition Prediction

> **Đồ án Khoa học Dữ liệu (Data Science Final Project - INFO3020)**  
> Phân tích nguyên nhân, đánh giá chất lượng dữ liệu và xây dựng mô hình dự báo nhân sự nghỉ việc tại tập đoàn IBM — **Nhóm 7**.

---

## 1. Giới thiệu dự án & Định hướng từ đầu (From Scratch Approach)

Dự án này được triển khai theo đúng chuẩn **vòng đời khoa học dữ liệu (Data Science Lifecycle)**, đi từ việc thiết lập bài toán kinh doanh cho đến khi xây dựng hoàn thiện mô hình dự báo nhân sự rời bỏ (`Attrition`). Toàn bộ quy trình được thực hiện một cách bài bản từ các bước cơ bản nhất:
* Khảo sát và xây dựng cấu trúc kho lưu trữ mã nguồn chuẩn mực (Clean Architecture).
* Khám phá dữ liệu sơ khai (EDA) kết hợp kiểm định chất lượng dữ liệu nghiêm ngặt.
* Xử lý triệt để các vấn đề về giá trị khuyết, nhiễu (Noise), và dữ liệu không nhất quán (`Consistency`).
* Xử lý ngoại lai (`Outliers`), biến đổi (`Transformation`) và thu gọn đặc trưng (`Reduction`) nhằm tối ưu hóa đầu vào cho mô hình học máy.

---

## 2. Mô tả bài toán (Business Understanding)

Bài toán đặt ra cho doanh nghiệp:
> Xây dựng mô hình phân tích và dự đoán khả năng nhân sự nghỉ việc (`Attrition = Yes/No`) dựa trên các thuộc tính về nhân khẩu học, thâm niên công tác, thu nhập, hiệu suất và mức độ hài lòng với môi trường làm việc.

### Mục tiêu cốt lõi
* **Phân tích nguyên nhân:** Tìm ra các yếu tố chủ chốt dẫn đến quyết định rời bỏ công ty của nhân viên.
* **Xây dựng hệ thống dự báo:** Huấn luyện các mô hình Machine Learning để nhận diện sớm nhân sự có nguy cơ nghỉ việc cao.
* **Đề xuất chiến lược quản trị:** Cung cấp cơ sở định lượng giúp ban lãnh đạo và phòng Nhân sự (HR) đưa ra các chính sách giữ chân nhân tài hiệu quả.

---

## 3. Tập dữ liệu (Dataset Overview)

* **Tên bộ dữ liệu:** IBM HR Analytics Employee Attrition & Performance
* **Nguồn gốc:** [Kaggle - IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
* **Quy mô:** 1,470 dòng $\times$ 35 cột thuộc tính.
* **Biến mục tiêu (Target):** `Attrition` (Yes / No).

### Các đặc trưng chính (Key Features)
* **Nhân khẩu học & Cá nhân:** `Age`, `Gender`, `Education`, `DistanceFromHome`, `MaritalStatus`.
* **Công việc & Thâm niên:** `Department`, `JobRole`, `JobLevel`, `YearsAtCompany`, `TotalWorkingYears`, `YearsSinceLastPromotion`.
* **Thu nhập & Chế độ:** `MonthlyIncome`, `MonthlyRate`, `PercentSalaryHike`, `StockOptionLevel`.
* **Đánh giá & Mức độ hài lòng:** `JobSatisfaction`, `EnvironmentSatisfaction`, `WorkLifeBalance`, `PerformanceRating`.

---

## 4. Vấn đề thực tiễn cần giải quyết

Tỷ lệ nghỉ việc cao gây ra tổn thất lớn về chi phí tuyển dụng, thời gian đào tạo và làm đứt gãy hoạt động vận hành của doanh nghiệp. Dự án tập trung giải đáp các câu hỏi trọng tâm:
* Nhân viên thuộc nhóm đối tượng nào có nguy cơ rời đi cao nhất?
* Các yếu tố như thu nhập (`MonthlyIncome`) hay thâm niên (`YearsAtCompany`) tác động ra sao?
* Làm thế nào để chuyển hóa dữ liệu nhân sự thô thành những thông tin hành động (Actionable Insights) cho doanh nghiệp?
