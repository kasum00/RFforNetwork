# STAGE 5.5 — Tối ưu RAM MI-RFE + RF (báo cáo đang cập nhật)

Trạng thái: IN_PROGRESS — benchmark kỹ thuật đang chạy tuần tự. Đây chưa phải kết quả chính thức FW512/FS8.

## Phạm vi

Bản runner riêng: D:\COMPACT-MI-RFE\huyen_code\ciciomt24_training_memorylab.py. Runner chính thức có SHA256 5E0FD78E8F306F65B38C5AF811FE8615C3AD6C278357DEB3C190963D608CAD5C trước thử nghiệm. Năm script CTU-13 và dataset không sửa.

Chỉ dùng FW512/FS16, mẫu cố định 8.192 Train (19 Benign proxy) và 2.048 Validation (5 Benign proxy). Test CSV không đọc dữ liệu, không predict/evaluate. Cả MI Top100, RF-RFE step1 và 100 giá trị k được giữ trong benchmark. Artifacts ở D:\COMPACT-MI-RFE\huyen_code\runs\stage55_memorylab.

## Số liệu đã có

| Chế độ | Peak RSS MiB | MI giây | RFE giây | k-search giây | Tổng giây | Best k | Tương đương |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline gốc | 239,301 | 16,532 | 41,512 | 103,088 | 162,272 | 1 | Tham chiếu |
| E1 | 239,602 | đang tổng hợp | đang tổng hợp | đang tổng hợp | 148,0 | 1 | MI/rank/100 confusion khớp sơ bộ |

Các số peak chỉ từ sample. Chênh RSS baseline–E1 dưới 1 MiB không chứng minh tiết kiệm. E2, E3, E4 và combined chưa có kết quả tại thời điểm ghi phần nháp.

## Kết luận admission hiện thời

Runner nguyên gốc FW512/FS8 vẫn ADMISSION_FAIL theo guard Stage5.4: cần 23,993641 GiB available, tổng RAM usable khoảng 23,730339 GiB. Optimized runner: INSUFFICIENT_EVIDENCE cho tới khi có đo và đánh giá đại diện; benchmark nhỏ không tự cấp PASS.

Báo cáo này sẽ được cập nhật theo artifacts hoàn tất. Nếu benchmark gián đoạn, trạng thái và artifacts dở vẫn được giữ, không trình bày như kết quả chính thức.
