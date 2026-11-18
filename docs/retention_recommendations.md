# Khuyến nghị giữ chân nhân sự (IBM HR Attrition – Nhóm 7)

Tỷ lệ nghỉ việc chung là **16,1%** (237/1.470). Các con số dưới đây lấy từ
`data/reports/eda_attrition_rate.csv` và `eda_numeric_summary.csv`
(chạy `python src/eda/eda_attrition.py`). Đây là **mối liên hệ thống kê, không phải quan hệ nhân quả**.

## 1. Phát hiện chính từ EDA

| Yếu tố | Nhóm rủi ro cao | Tỷ lệ nghỉ việc | So sánh |
|---|---|---|---|
| Làm thêm giờ | Có OverTime (n=416) | 30,5% | Không OverTime: 10,4% (gấp ~2,9 lần) |
| Đi công tác | Travel_Frequently (n=277) | 24,9% | Non-Travel: 8,0% |
| Quyền chọn cổ phiếu | Mức 0 (n=631) | 24,4% | Mức 1: 9,4%; mức 2: 7,6% |
| Tình trạng hôn nhân | Single (n=470) | 25,5% | Married 12,5%; Divorced 10,1% |
| Thu nhập tháng | Dưới 3.000 USD | 28,6% | Trên 12.000 USD: 5,6% |
| Vị trí | Sales Representative (n=83) | 39,8% | Laboratory Technician 23,9%; HR 23,1% |
| Khoảng cách đi làm | Trên 10 km | 20,0–22,1% | Từ 1–5 km: 13,8% |
| Hài lòng công việc | Mức 1 (n=289) | 22,8% | Mức 4: 11,3% |
| Hài lòng môi trường | Mức 1 (n=284) | 25,4% | Mức 3–4: ~13,5% |
| Cân bằng công việc | Mức 1 (n=80) | 31,2% | Mức 3: 14,2% |
| Thâm niên | Trung vị nhóm nghỉ việc: 3 năm | – | Nhóm ở lại: 6 năm |
| Thu nhập trung vị | Nhóm nghỉ việc: 3.202 USD | – | Nhóm ở lại: 5.204 USD |

Lưu ý: nhóm "Cân bằng công việc mức 1" và "Sales Representative" có cỡ mẫu nhỏ, nên cần thận trọng khi diễn giải.

## 2. Hành động đề xuất

1. **Kiểm soát làm thêm giờ.** OverTime là yếu tố có chênh lệch lớn nhất và cũng nằm trong nhóm đặc trưng quan trọng của mô hình (hạng 2 trong bảng xếp hạng đồng thuận). Đặt ngưỡng giờ làm thêm, yêu cầu quản lý rà soát phân bổ công việc khi một nhân viên làm thêm giờ liên tục.
2. **Rà soát lương cho nhóm thu nhập thấp.** Nhóm dưới 3.000 USD/tháng nghỉ việc gần gấp đôi mức trung bình. Ưu tiên hiệu chỉnh lương cho các vị trí kỹ thuật và bán hàng cấp thấp.
3. **Mở rộng quyền chọn cổ phiếu / phúc lợi dài hạn.** Nhân viên không có cổ phiếu (mức 0) nghỉ việc nhiều hơn khoảng 2,6 lần nhóm mức 1. Cần kiểm tra thêm vì mức 0 cũng tập trung ở nhân viên mới.
4. **Theo dõi nhân viên trẻ, thâm niên thấp, độc thân.** Nhóm nghỉ việc có tuổi trung vị 32 và thâm niên 3 năm. Gợi ý: trao đổi định kỳ trong 3 năm đầu và lộ trình phát triển rõ ràng.
5. **Giảm tải công tác cho nhóm đi công tác thường xuyên.** Xem xét luân phiên, bù thời gian nghỉ, hoặc họp từ xa khi có thể.
6. **Ưu tiên Sales Representative, Laboratory Technician, HR.** Đây là các vị trí có tỷ lệ nghỉ việc cao nhất; nên phỏng vấn nghỉ việc và khảo sát hài lòng theo vị trí.
7. **Hỗ trợ nhân viên ở xa (trên 10 km).** Phụ cấp đi lại, xe đưa đón hoặc lịch làm việc linh hoạt/làm việc từ xa một phần.
8. **Cải thiện môi trường và cân bằng công việc – cuộc sống** tại các đội có điểm hài lòng ở mức 1.

## 3. Cách dùng mô hình dự báo

Mô hình chọn: Logistic Regression có class weight + SMOTE, ngưỡng 0,60 (xem `data/reports/model_comparison.csv`).
Trên tập test, mô hình phát hiện khoảng 69% số người nghỉ việc (recall) nhưng chỉ khoảng 45% trong số người bị gắn cờ thực sự nghỉ việc (precision), tức là cứ 2 người bị gắn cờ thì có khoảng 1 báo động nhầm.

- Dùng điểm rủi ro để **ưu tiên trò chuyện giữ chân**, không để đánh giá, kỷ luật hay quyết định tuyển dụng/sa thải.
- Tập test nhỏ (~221 người, ~36 người nghỉ việc) nên chỉ số có thể dao động đáng kể; nên theo dõi thêm bằng cross-validation và dữ liệu mới.

## 4. KPI theo dõi
- Tỷ lệ nghỉ việc theo phòng ban, vị trí và nhóm OverTime
- Tỷ lệ nhân viên rủi ro cao được quản lý trao đổi trong vòng 30 ngày
- Điểm hài lòng (công việc, môi trường, cân bằng) theo quý
- Tỷ lệ hoàn thành rà soát lương và thăng tiến
