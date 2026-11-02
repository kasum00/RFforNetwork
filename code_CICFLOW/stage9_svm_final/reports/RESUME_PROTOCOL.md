# Resume và Test exactly once

## Final fit chưa có Test claim

1. Kết nối Drive, checkout đúng Stage9 commit và các selection saved commit. Không pull/reset source của run đã tạo.
2. Cell1 giữ đúng METHOD, selection run ID và Stage9 run ID. Chạy Cells2–10.
3. Nếu `FINAL_COMPLETE.json` verified: dùng Cell13, không cần fit lại.
4. Nếu chỉ local `fit_*/final_checkpoint.json` hoàn chỉnh: `resume-final` xác minh và publication bundle đó; không SVC.fit lại.
5. Nếu remote final_model đã đủ bundle verified, chỉ thiếu completion marker: `resume-final` xác minh rồi tạo marker; không fit lại.
6. Nếu SVC.fit bị ngắt trước checkpoint hoàn chỉnh và chưa có remote bundle/claim: `resume-final` có thể tính lại final fit, vẫn cùng features/params/full rows; không chạy MI/RFE/k-search.
7. Nếu remote bundle partial/corrupt, hoặc mất local bundle cần cho publication: BLOCKED, bảo tồn toàn bộ để kiểm tra. Không tự xóa/ghi đè file.

Lease chưa release không chứng minh worker còn chạy hoặc đã chết. Chỉ sau khi biết worker cũ thực sự dừng mới điền ở Cell1:

```python
PREVIOUS_WORKER_STOPPED[(METHOD, 'FW256_FS8')] = 'PREVIOUS STAGE9 WORKER STOPPED <token trong blocker>'
```

Không điền khi không chắc. Token này chỉ xử lý quyền worker; không bao giờ bỏ chặn claim Test.

## Test

| Durable state | Hành động |
|---|---|
| Chưa claim + final verified + xác nhận đúng | Exclusive claim/readback, rồi mới hash/footer/read Test/predict một lần |
| Test completion verified | Đọc artifacts, recompute metrics từ saved y/prediction; không Test dataset/predict lại |
| Claim tồn tại, completion thiếu/hỏng/không rõ | BLOCKED; không auto retry hoặc refit model |
| Prediction local đã có nhưng publication bị ngắt | Giữ nguyên local/remote; manual publication recovery phải được thiết kế/xác minh riêng, không evaluate lại |
| Native exit0 nhưng không có verified marker | Không kết luận completion/PASS |

Claim đặt ở namespace cố định method/config, trước mọi byte Test kể cả hashing; đổi Stage9 run_id không tạo một Test budget mới. Không claim chung giữa A1 và B hoặc giữa các config.

Test ledger gồm immutable `TEST_CLAIM.json`, completion `test_ledger.json`, `test_checkpoint.json`, `TEST_COMPLETE.json`; trạng thái effective dựa vào đầy đủ chain, không sửa file claim thành COMPLETE.

Không có mid-fit checkpoint libsvm. Dừng monitor chỉ dừng việc theo dõi; worker có thể vẫn chạy. Không tắt runtime/đổi thư viện/source trong lúc final fit hoặc Test. Nếu runtime ngắt sau claim, áp dụng trạng thái BLOCKED ở trên.
