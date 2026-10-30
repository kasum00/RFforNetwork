# Chạy B step2 trên Google Colab

Chưa có lệnh nào được Codex thực thi. Các bước dưới do bạn chủ động chạy.

## 1. Đưa source B lên GitHub

Chỉ stage package B và notebook mới. Giữ checkout/source A1 đang chạy nguyên trạng.
Trong PowerShell tại máy Windows của bạn:

~~~powershell
Set-Location -LiteralPath 'D:\COMPACT-MI-RFE'
git add -- code_CICFLOW/phuong_anB_step2 notebooks/PHUONG_AN_B_MI_RFE_SVM_STEP2_COLAB.ipynb
git diff --cached --stat -- code_CICFLOW/phuong_anB_step2 notebooks/PHUONG_AN_B_MI_RFE_SVM_STEP2_COLAB.ipynb
git commit --only -m "Add isolated CICFLOW B step2 coarse-to-fine SVM selection" -- code_CICFLOW/phuong_anB_step2 notebooks/PHUONG_AN_B_MI_RFE_SVM_STEP2_COLAB.ipynb
git push -u origin HEAD
git rev-parse HEAD
~~~

Copy SHA40 cuối. Đây phải là commit mới có B, không phải SHA A1 38e182….
Điền SHA vào bản notebook đang mở trên Colab; không cần commit lại source chỉ để
điền SHA trong bản giao diện Colab. Checkout B sẽ nằm riêng ở /content/CICFLOW_B_STEP2_<SHA12>.

Nếu dependencies/helper trong checkout mới không trùng các hash frozen đã review,
preflight BLOCKED. Không sửa approved_helpers.json để ép PASS; phải đối chiếu source.

## 2. Chạy từng cell

1. Cell 1: chỉ bật PREPARE_ENVIRONMENT=True nếu cần cài đúng profile. Sau cài chọn
   Runtime → Restart session rồi đặt cờ False. Không thay library khi worker còn chạy.
2. Cell 2: mount MyDrive. Nếu credential propagation thất bại, hoàn tất xác thực
   Google/Colab rồi chạy lại; không chạy training hoặc tạo output giả khi chưa mount.
3. Cell 3: kiểm tra đúng DATASET_ROOT=DRIVE/CICFLOWMETER_FEATURE_CSV; chọn config,
   Run ID RUN_B2_01, step=2 và SHA mới. Mặc định SELECTED_CONFIGS chỉ có FW256_FS8.
   Policy đã khóa theo source bạn cung cấp: COARSE=(10,20,30,50,75,100), RADIUS=5,
   SEARCH_POLICY=phuong_anB_coarse_to_fine_v1. RFE_STEP=2 độc lập coarse grid.
4. Cell 4: clone/check clean source và environment. Không pull/reset checkout cũ.
5. Cell 5: kiểm tra dataset sidecar identity đủ bốn config. Đây là metadata check,
   chưa phải xác minh Parquet values/full preflight PASS.
6. Cell 6: chạy synthetic selftest. Native0 cùng evidence.status=PASS, không skipped,
   source/environment đúng mới được dùng làm evidence cho preflight.
7. Cell 7: xem policy gốc và chạy preflight/queue plan. Không có yêu cầu chọn hoặc
   phê duyệt policy thay thế. Source/identity/checkpoint mismatch vẫn BLOCKED.
8. Cell 8: nhập đúng từng câu RUN/RESUME EXPERIMENTAL B STEP2 <CONFIG>/<RUN_ID> được in.
   Chưa có worker ở bước này.
9. Cell 9: đặt EXECUTE_QUEUE=True rồi chạy. Queue tuần tự và tự verify sau mỗi config
   complete. Cell trả quyền điều khiển không có nghĩa worker đã xong.
10. Cell 10: monitor tqdm.auto; progress là whole durable units, current k, phase,
    best score, CPU/RAM/elapsed/ETA đo thực tế. Config chưa chạy hiện WAITING.
11. Cell 11: REQUEST_STOP=True để pause ở safe boundary, hoặc STOP_AFTER_CONFIG=True
    để config hiện tại hoàn tất/verify rồi không chạy config kế tiếp.
12. Cell 12: PREPARE_RESUME=True chỉ sau khi worker dừng. Giữ source/data/environment/
    step/policy/Run ID. Sau đó chạy lại Cell 7–8–9. Lease chưa release phải được xử
    lý với actual ID và xác nhận runtime cũ thực sự dừng; không sửa lease.
13. Cell 13: xem best k, Validation Accuracy, RF/SVM counts và coarse/fine/session times.
    SAVED_REPORT_ONLY chưa phải SHA verify mới.
14. Cell 14: VERIFY_ARTIFACTS=True nếu muốn verify lại completed selection riêng.
    Không chạy trong lúc queue còn active. Không có final refit hoặc Test command.

Muốn chạy cả bốn tuần tự, tại Cell 3 đặt SELECTED_CONFIGS=list(CONFIGS). Muốn chạy
lần lượt từng cấu hình, mỗi lần để danh sách một phần tử và chạy lại preflight/
confirmation/queue sau khi JOB.running=False. Đổi config không sửa batch đã bắt đầu.

Mỗi config mặc định cùng RUN_B2_01 nhưng nằm trong thư mục config riêng. Không tạo
Run ID mới để né mismatch. Run complete AUTO→verify, run đang dở AUTO→resume,
chưa có run AUTO→run. A1 RUN_A1_* không được dùng cho B.

## 3. Lệnh unit/regression test Windows có logging

Dùng Python có đúng profile dependencies. Command này gọi selftest guard, tạo
JSON evidence unique và dùng logging wrapper đã có; không gọi training:

~~~powershell
Set-Location -LiteralPath 'D:\COMPACT-MI-RFE\code_CICFLOW'
$b2Evidence = Join-Path 'D:\COMPACT-MI-RFE\code_CICFLOW\logs' ('B2_tests_' + [guid]::NewGuid().ToString('N') + '.json')
$b2Command = 'python -B -u -m phuong_anB_step2.selftest --source "D:\COMPACT-MI-RFE\code_CICFLOW" --evidence "' + $b2Evidence + '"'
& 'D:\COMPACT-MI-RFE\code_CICFLOW\shared\Invoke-CicflowLogged.ps1' -Stage 'B STEP2' -Task 'synthetic-unit-tests' -Experiment 'phuong_anB_step2' -Configuration 'SYNTHETIC' -RunId 'UNIT_B2' -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' -OutputDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW\logs' -ScientificStatus 'NOT_ASSESSED' -Command $b2Command
Write-Host "NATIVE EXIT: $LASTEXITCODE"
Write-Host "EVIDENCE: $b2Evidence"
~~~

Evidence Windows không thay evidence Colab: môi trường/source phải trùng runtime
thực nghiệm. Trước training Colab vẫn chạy Cell 6 trong checkout B đã pin.

Unit discovery tương đương (khi cần debug từng assertion, không tạo evidence admission):

~~~powershell
& 'D:\COMPACT-MI-RFE\code_CICFLOW\shared\Invoke-CicflowLogged.ps1' -Stage 'B STEP2' -Task 'unittest-discovery' -Experiment 'phuong_anB_step2' -Configuration 'SYNTHETIC' -RunId 'UNIT_B2' -WorkingDirectory 'D:\COMPACT-MI-RFE\code_CICFLOW' -ScientificStatus 'NOT_ASSESSED' -Command 'python -B -u -m unittest discover -s phuong_anB_step2/tests -t . -p "test_*.py" -v'
~~~

## 4. Lệnh Colab theo từng config, có logging tự động

Sau setup/tests/preflight, dùng launcher thay vì gọi worker CLI trần:

~~~python
# Preflight bốn cấu hình; không fit.
for config in CONFIGS:
    result = launch(request_for(config, 'preflight'))
    print(config, result['native_exit_code'], result['status'])
~~~

Full/resume nên dùng Cell 7–9 để có queue và monitor. Nếu chủ động muốn worker
foreground cho một config, thao tác sau vẫn có exact confirmation, gate và log:

~~~python
config = 'FW256_FS8'  # hoặc FW256_FS16, FW512_FS8, FW512_FS16
mode = 'run'          # 'resume' cho đúng B run đã tồn tại
request = request_for(config, mode)
exact = confirmation(request)
if input(exact + '\n') != exact:
    raise RuntimeError('Confirmation differs')
request['confirmation'] = exact
result = launch(request)
print(json.dumps(result, indent=2))
~~~

Chỉ gọi khi queue/worker khác không active. Foreground này chặn cell cho tới worker
kết thúc; queue ở Cell 9 là cách thuận tiện để dùng monitor riêng.

Verify completed artifacts của đủ bốn config:

~~~python
for config in CONFIGS:
    result = launch(request_for(config, 'verify'))
    print(config, result['native_exit_code'], result['status'], result.get('phase_status'))
~~~

Config chưa complete sẽ BLOCKED, không tự train. Mọi result giữ native exit và
program/scientific status riêng; không suy ra official reproduction từ PASS.

## 5. Các file kết quả

Drive: MyDrive/experiment_results/phuong_anB_step2/mi_rfe_svm/<CONFIG>/<RUN_ID>/.
Giữ toàn bộ run để verify, gồm checkpoints và manifests, không chỉ results.

Trong results:
- selection.json: best k, selected features, full Validation metrics, scientific status.
- k_search_results.json: coarse/fine grids, policy, từng measured row và winner.
- features.json, rfe_feature_ranking.csv, mi_top100.json, mi_feature_scores.csv.
- fine_plan.json: refinement grid và digest coarse measurements.
- statistics.json: fit counts, phase times, recorded wall-time scope, benchmark_speedup=None.
- selection_frozen.json, selection_report.md, artifact_manifest.json, SELECTION_COMPLETE.json.

Log tổng hợp Colab: MyDrive/code_CICFLOW/logs/ALL_EXECUTION_RESULTS.log.
Raw stdout/stderr được gộp có nhãn, append trong lúc chạy; local log + native receipt
ở local run/controller_logs, còn worker log được bind vào từng Drive checkpoint.
Ghi log lỗi có warning và flag, không tuyên bố đã lưu đủ. Hard runtime loss có thể
mất output local chưa persist; không bịa lịch sử.

## Đối chiếu chính sách gốc

Đã dùng fine_grid/choose_best từ run_phuong_anB.py do bạn dán, kèm regression test
so sánh trực tiếp pure helpers. Thay đổi thuật toán B gốc chỉ ở RF-RFE step1→2.
Source/format mới v2 không resume checkpoint draft grid cách2 hoặc ranking step1.
Không sửa/xóa run cũ để vượt gate. Không tái sử dụng A1 checkpoints cho B.
