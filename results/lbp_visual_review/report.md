# Đánh giá trực quan trước khi loại bỏ LBP 28D

## Phạm vi

- Chỉ chấm mức tương đồng thị giác, không dùng tên loài hay nhãn để cho điểm.
- Thang điểm 1–5 cho màu sắc, vân/bề mặt lông, hình dáng và tổng thể.
- Hai lớp kiểm tra:
  1. 6 truy vấn đại diện, chấm toàn bộ 60 kết quả Top-10.
  2. 6 truy vấn nhạy với LBP nhất, so trực tiếp hai ảnh duy nhất bị hoán đổi khỏi/vào Top-10.

## Kết quả Top-10 đại diện

Trong cả 6 truy vấn đại diện, `Current Full` và `Current Full - LBP` trả về đúng cùng 10 ảnh, cùng thứ tự. Vì vậy điểm trực quan của hai cấu hình bằng nhau tuyệt đối trên tập này.

| Chỉ tiêu trung bình | Điểm |
|---|---:|
| Màu sắc | 3.500 / 5 |
| Vân/bề mặt lông | 3.317 / 5 |
| Hình dáng | 2.850 / 5 |
| Tổng thể | 3.033 / 5 |
| Tỷ lệ kết quả có điểm tổng thể >= 4 | 35.0% |

Điểm chi tiết nằm trong `visual_scores.csv`.

## Mức độ LBP thật sự làm thay đổi Top-10

Đã so toàn bộ 840 truy vấn:

| Chỉ tiêu | Kết quả |
|---|---:|
| Top-10 giống cả tập lẫn thứ tự | 707 / 840 (84.17%) |
| Top-10 có cùng tập ảnh | 801 / 840 (95.36%) |
| Chỉ khác đúng 1 ảnh trong Top-10 | 39 / 840 (4.64%) |
| Khác từ 2 ảnh trở lên | 0 / 840 |
| Số ảnh trùng Top-10 trung bình | 9.954 / 10 |
| RBO trung bình giữa hai cấu hình | 0.9965 |

## Chấm các trường hợp nhạy nhất

Với mỗi trường hợp, chỉ chấm cặp ảnh duy nhất bị LBP làm thay đổi. Đây là nơi có khả năng quan sát đóng góp của LBP rõ nhất.

| Cấu hình tạo ảnh được đưa vào Top-10 | Điểm vân trung bình | Điểm tổng thể trung bình | Số lần được chọn tốt hơn |
|---|---:|---:|---:|
| Giữ LBP 28D | 1.833 / 5 | 1.667 / 5 | 1 / 6 |
| Bỏ LBP 28D | 3.333 / 5 | 3.000 / 5 | 5 / 6 |

Ảnh đối chiếu nằm trong `sensitive_cases/sensitive_*.png`; điểm và ghi chú nằm trong `sensitive_cases/sensitive_unique_scores.csv`.

## Kết luận

**Khuyến nghị: loại bỏ LBP 28D hiện tại khỏi production.**

Lý do:

- LBP hiện tại không cải thiện chất lượng vân lông quan sát được trên Top-10 đại diện.
- Khi LBP thực sự làm thay đổi tập kết quả, phương án không LBP được đánh giá tốt hơn ở 5/6 trường hợp nhạy nhất.
- Metric retrieval tổng thể gần như không đổi khi bỏ LBP, trong khi thời gian trích xuất giảm khoảng 84 ms/truy vấn theo benchmark hiện có.
- Kết luận này chỉ áp dụng cho **LBP global uniform 28D hiện tại**, không có nghĩa mọi biến thể LBP đều vô ích. Spatial/multi-scale LBP có thể được nghiên cứu riêng nhưng kết quả trước đó chưa đủ tốt để đưa vào production.

Production chưa được chỉnh sửa trong bước đánh giá này.
