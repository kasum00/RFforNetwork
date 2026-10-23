# MI-RFE SVM: chạy tiếp trên Windows VM

Trạng thái: IMPLEMENTED / NOT EXECUTED. Codex không chạy Python, tests, preflight, cài package hoặc training; không commit/push. Không sửa checkpoint/dataset thật.

## 1. Cách dùng trong trường hợp của bạn

Checkpoint và dataset đã có trên máy ảo: KHÔNG cần tải/copy lại. Bạn chỉ:

1. Đẩy source mới từ máy thật lên GitHub bằng danh sách file bên dưới.
2. Clone/cập nhật source trên máy ảo; dùng Python 3.13 x64 và môi trường riêng.
3. Điền đúng đường dẫn các thư mục sẵn có vào config ngoài Git.
4. Tự chạy tests → inspect → preflight → duyệt ngân sách/approval → run.

Notebook Colab không còn là điểm vào Windows. CLI mới: local.run_mi_rfe_svm. Chỉ hỗ trợ MI-RFE SVM, FW_512_FS_16 và run mi_rfe_svm_FW512_FS16_resumable01 bắt nguồn từ commit 7718c74c44070600ea876a3a1c678632ccb184dd. Không có nhánh âm thầm tạo run mới/chạy từ k=1.

Repository thực tế đã đọc trên máy đang sửa: D:\COMPACT-MI-RFE. Đường dẫn C:\Users\LAB_LEE\Desktop\RFforNetwork_GITHUB trong ví dụ là đường dẫn clone đề xuất trên VM; lúc kiểm tra máy thật, đường dẫn LAB_LEE này không tồn tại. Đổi biến taskRepo trên VM nếu khác.

## 2. Cấu trúc thư mục cần thiết

Code clone và dữ liệu/checkpoint có thể ở hai ổ khác nhau.

~~~
C:\Users\LAB_LEE\Desktop\RFforNetwork_GITHUB\
  .git\
  .venv\                         tạo riêng trên VM; không đưa lên Git
  CODE_ctu13\                    5 script gốc, giữ nguyên byte
  huyen_code\                    scientific core hiện có, không thay thuật toán
  colab\                         checkpoint store/engine dùng chung
  configs\cplus_manifest\
  local\
    run_mi_rfe_svm.py
    paths.py
    checkpoint_migration.py
    windows_preflight.py
    local_authorization.py
    local_supervisor.py
    windows_job.py
    progress_console.py
    test_windows_local.py
    test_local_supervisor.py
    requirements-windows.txt
    windows_paths.example.json
    local_budget.BLOCKED.example.json
    README_WINDOWS_LOCAL.md
    WINDOWS_LOCAL_REPORT.md

D:\COMPACT-MI-RFE\
  CICIoMT24\
    WINDOWS_CPLUS\
      window_run_summary.json
      FW_256_FS_8\train|val|test\scenario\..._windows.csv
      FW_256_FS_16\train|val|test\scenario\..._windows.csv
      FW_512_FS_8\train|val|test\scenario\..._windows.csv
      FW_512_FS_16\train|val|test\scenario\..._windows.csv
    CSV_SPLIT_CPLUS\
      split_manifest.csv
      validation.json
  approvals\
    mi_rfe_svm_FW512_FS16_resumable01.session01.json   approval Colab gốc
  technical_probes\
    <file probe thực sự đã được approval session01 chấp thuận>.json
  environment\
    ciciomt24_windows_cplus_ssd_file_sha256_manifest.json
  svm_resumable\mi_rfe_svm\FW_512_FS_16\
    mi_rfe_svm_FW512_FS16_resumable01\               chính là checkpoint-root
      run_manifest.json
      active_session.json                          có thể còn do Colab ngắt; KHÔNG tự xóa
      progress.json
      events.jsonl
      heartbeat.jsonl
      .session_claim.guard                         giữ nếu có
      checkpoints\
        data_prepared\data.json + checkpoint.json
        mi\mi.joblib + top100_scaler.joblib + mi_feature_scores.csv + data.json + checkpoint.json
        rfe\rfe_feature_ranking.csv + data.json + checkpoint.json
        k_search\
          k_001\data.json + checkpoint.json
          ...
          k_075\data.json + checkpoint.json         chỉ minh họa; scanner xác định thực tế
          k_076.incomplete-...\                     giữ nếu có, không coi COMPLETE
        selection\                                 nếu đã hoàn thành
        final\                                     nếu đã hoàn thành
        final_test\                                nếu đã hoàn thành
      sessions\
        session01\
          authorization.json
          preflight.json
          session_result.json                      có thể thiếu nếu Colab chết đột ngột
        session02\                                 chỉ tạo sau approval khi run
          authorization.json
          local_execution.json
          local_preflight.json
          job_assignment.json
          supervisor_progress.json
          local_worker_result.json
          session_result.json
      results\
        test_evaluation_claim.json                 nếu có mà final_test chưa xong: BLOCK
        completed\                                 kết quả cuối được xác minh
  windows_local_control\                           ngoài source/ngoài checkpoint
    windows_paths.json
    local_budget.json
    unit_tests.json
    session02.preflight01.json
    session02.approval.json
  windows_local_runs\                              output-root: cache, không phải run mới
    resumable_cache\<origin-binding-sha>\
      train\  val\  top100\
~~~

Tên thư mục vật lý không bắt buộc. checkpoint-root phải trỏ tới folder chứa run_manifest.json, KHÔNG trỏ tới folder checkpoints/ hoặc folder svm_resumable/ chung.

WindowStore hiện có yêu cầu đủ inventory 322 CSV thuộc cả bốn FW/FS để giữ fingerprint gốc; chỉ huấn luyện cấu hình 512/16. Nếu VM chỉ có riêng FW_512_FS_16, runner sẽ BLOCK chứ không tự đổi fingerprint.

## 3. Push từ máy thật — tự chạy, Codex không thực hiện

~~~powershell
Set-Location 'D:\COMPACT-MI-RFE'
git status --short
git diff -- .gitignore .gitattributes colab/svm_resumable_training.py colab/test_svm_resumable_training.py
git add -- local .gitignore .gitattributes README.md colab/svm_resumable_training.py colab/test_svm_resumable_training.py
git diff --cached --name-only
git diff --cached --stat
# Chỉ sau khi tự review danh sách và diff:
git commit -m "Add verified Windows local continuation for MI-RFE SVM"
git push origin HEAD:main
git rev-parse HEAD
~~~

Không git add .; không commit dataset, approvals hiệu lực, manifests checkpoint, model, cache hay log. Kiểm tra staged có thay đổi của công việc khác không. Codex không reset/stash hoặc xử lý các thay đổi khác.

## 4. Clone trên VM và khóa line ending

Nếu đã có clone sạch mới đúng source thì không cần clone thêm. Để giữ SHA256 byte của CTU-13, dùng core.autocrlf=false và .gitattributes mới. Không chỉnh scientific hash để bỏ qua lỗi CRLF.

~~~powershell
$taskRepo = 'C:\Users\LAB_LEE\Desktop\RFforNetwork_GITHUB'
if (Test-Path -LiteralPath $taskRepo) { throw 'Clone da ton tai; kiem tra git status/commit truoc, khong ghi de.' }
git -c core.autocrlf=false clone 'https://github.com/kasum00/RFforNetwork.git' $taskRepo
git -C $taskRepo config core.autocrlf false
Set-Location $taskRepo
git status --short
git log -1 --oneline
git cat-file -t '7718c74c44070600ea876a3a1c678632ccb184dd'
~~~

Không clone --depth 1: migration cần đọc blob của commit nguồn. Nếu history thiếu, tự fetch trước, không sửa checkpoint:

~~~powershell
git -C 'C:\Users\LAB_LEE\Desktop\RFforNetwork_GITHUB' fetch origin
~~~

Nếu clone cũ đã chuyển byte CTU-13 thì dừng, lưu thay đổi cá nhân rồi clone sạch sang thư mục mới; không git reset --hard. Manifest dữ liệu phải là bản byte-identical đã dùng ở Colab, không tự convert newline.

## 5. CPU/RAM/SSD và môi trường

Các lệnh này là để BẠN tự chạy trên VM, không phải thông số đã đo.

~~~powershell
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory
Get-Volume -DriveLetter C,D | Select-Object DriveLetter,Size,SizeRemaining
py -0p
py -3.13 --version
~~~

Giữ đúng Python 3.13 x64 và profile đã dùng trên Colab. Chưa xác minh wheel availability từ môi trường hiện tại. Cài bằng only-binary để nếu thiếu wheel thì DỪNG, không tự build/thay phiên bản.

~~~powershell
$taskRepo = 'C:\Users\LAB_LEE\Desktop\RFforNetwork_GITHUB'
py -3.13 -m venv "$taskRepo\.venv"
$taskPython = "$taskRepo\.venv\Scripts\python.exe"
& $taskPython -m pip install --only-binary=:all: -r "$taskRepo\local\requirements-windows.txt"
if ($LASTEXITCODE -ne 0) { throw 'Khong cai duoc dung binary profile; dung va gui log, khong tu doi versions.' }
& $taskPython -c "import sys,struct,numpy,pandas,scipy,sklearn,joblib,psutil; print(sys.version,struct.calcsize('P')*8); print(numpy.__version__,pandas.__version__,scipy.__version__,sklearn.__version__,joblib.__version__,psutil.__version__)"
~~~

SVM không được chuyển GPU/cuML hoặc LinearSVC. Không giảm cache_size/mẫu/k_grid để vừa RAM.

## 6. Điền config và budget ngoài Git

Copy hai template CHỈ khi chưa có file đích, rồi sửa thủ công trong VS Code. Nếu đã cài đặt trên VM như bạn nói, chỉ sửa config hiện có cho đúng, không copy lại dữ liệu.

~~~powershell
$taskRepo = 'C:\Users\LAB_LEE\Desktop\RFforNetwork_GITHUB'
$taskControl = 'D:\COMPACT-MI-RFE\windows_local_control'
New-Item -ItemType Directory -Path $taskControl -Force | Out-Null
if (Test-Path -LiteralPath "$taskControl\windows_paths.json") { throw 'Config da ton tai; sua thu cong, khong ghi de.' }
Copy-Item -LiteralPath "$taskRepo\local\windows_paths.example.json" -Destination "$taskControl\windows_paths.json"
if (Test-Path -LiteralPath "$taskControl\local_budget.json") { throw 'Budget da ton tai; sua thu cong, khong ghi de.' }
Copy-Item -LiteralPath "$taskRepo\local\local_budget.BLOCKED.example.json" -Destination "$taskControl\local_budget.json"
~~~

Trong windows_paths.json, sửa repo theo clone VM, đủ đường dẫn tuyệt đối, technical_evidence theo file thật. Dataset/checkpoint-root giữ nơi đã cài sẵn.

local_budget.json cố ý có null: chưa được cấp quyền chạy. BẠN điền số byte nguyên dương sau khi xem tài nguyên VM, rồi xem xét lại report:

- max_elapsed_seconds: mặc định 36000; >36000 cần xác nhận riêng khi approve.
- max_tree_rss_bytes: giới hạn tổng RSS worker + con, không phải giới hạn RAM của host.
- min_start_available_ram_bytes: ngưỡng RAM available ban đầu do bạn duyệt, không dùng mặc định 42 GiB.
- min_runtime_available_ram_bytes: phần RAM phải còn cho hệ điều hành.
- min_free_disk_bytes: dung lượng trống tối thiểu ở cả volume checkpoint và output/cache.
- poll_interval_seconds: 1..60, đề xuất 10.

Ví dụ 1 GiB = 1073741824 byte. Không lấy một budget ví dụ làm đo đạc thực tế. Guard còn yêu cầu RSS limit + reserve <= RAM available và ước lượng bảo thủ mảng/cache SVM <= RSS limit, <=85% RAM available. Không đủ thì dừng, không giảm dữ liệu.

## 7. Tests và inspect — bạn tự chạy

~~~powershell
$taskRepo = 'C:\Users\LAB_LEE\Desktop\RFforNetwork_GITHUB'
$taskPython = "$taskRepo\.venv\Scripts\python.exe"
Set-Location $taskRepo
$env:PYTHONPATH = "$taskRepo;$taskRepo\huyen_code"
$env:PYTHONUTF8 = '1'
& $taskPython -m local.run_mi_rfe_svm tests --repo $taskRepo --output 'D:\COMPACT-MI-RFE\windows_local_control\unit_tests.json'
if ($LASTEXITCODE -ne 0) { throw 'Unit tests FAIL; training van BLOCKED.' }
& $taskPython -m local.run_mi_rfe_svm inspect --config-file 'D:\COMPACT-MI-RFE\windows_local_control\windows_paths.json'
~~~

tests subcommand CHỈ chạy khi bạn gọi, chạy bảy suite synthetic/mocks và ghi evidence commit/source sau exit 0; không dùng benchmark hoặc full dataset. Evidence không overwrite: source/commit đổi cần file evidence mới và sửa config.

inspect không fit/unpickle: kiểm binding, byte/SHA artifact, MI/RFE, thứ tự/prefix k, khoảng trống, sample counts và Test claim. Output next_stage/next_k là kết quả thực tế phải dùng; không mặc định k76. Nếu k76 có folder tên chính thức mà chưa checkpoint.json/corrupt, BLOCK. Folder .incomplete-* giữ nguyên nhưng không được tính hoàn tất. Nếu không có k nào hoàn thành, BLOCK thay vì từ k1.

## 8. Preflight → approval → run

~~~powershell
$taskRepo = 'C:\Users\LAB_LEE\Desktop\RFforNetwork_GITHUB'
$taskPython = "$taskRepo\.venv\Scripts\python.exe"
Set-Location $taskRepo
$env:PYTHONPATH = "$taskRepo;$taskRepo\huyen_code"
$taskConfig = 'D:\COMPACT-MI-RFE\windows_local_control\windows_paths.json'
$taskReview = 'D:\COMPACT-MI-RFE\windows_local_control\session02.preflight01.json'
$taskApproval = 'D:\COMPACT-MI-RFE\windows_local_control\session02.approval.json'
& $taskPython -m local.run_mi_rfe_svm preflight --config-file $taskConfig --report $taskReview
if ($LASTEXITCODE -ne 0) { throw 'MIGRATION/RESOURCE BLOCKED; doc report, khong train.' }
Get-Content -LiteralPath $taskReview
# Chỉ khi bạn đồng ý nguồn/checkpoint, tài nguyên và cross-OS limitations:
& $taskPython -m local.run_mi_rfe_svm approve --config-file $taskConfig --preflight-report $taskReview --approval $taskApproval
if ($LASTEXITCODE -ne 0) { throw 'Chua co Windows approval hop le.' }
$taskPhrase = (Get-Content -LiteralPath $taskApproval -Raw | ConvertFrom-Json).confirmation
& $taskPython -u -m local.run_mi_rfe_svm run --config-file $taskConfig --preflight-report $taskReview --approval $taskApproval --confirmation $taskPhrase
~~~

approve yêu cầu nhập chính xác các câu xác nhận: source/run/session, Colab session01 đã dừng, cross-OS checkpoint reuse và single Test. Nếu còn lock cũ, thêm xác nhận đúng previous session. Nếu heartbeat cũ còn mới hoặc worker local cũ còn sống thì run BLOCK dù đã xác nhận.

Preflight hash Train/Val; worker hash lại Train/Val trước sử dụng. Test: trước final chỉ stat/inventory, không đọc header/nội dung; sau final model đã commit và Test đã claim thì hash byte Test theo checksum gốc rồi mới load/predict một lần. SHA mismatch tại bước này giữ claim và BLOCK mọi retry tự động.

Một số check preflight phải đọc toàn bộ bytes Train/Val: không phải audit lại flow/sliding window và không fit, nhưng sẽ tốn thời gian I/O. Không xóa evidence/manifest để lách gate.

## 9. Tiến độ, stop và phiên tiếp theo

Worker dùng terminal stdout/stderr trực tiếp, UTF-8/unbuffered; event fit/checkpoint in ngay, heartbeat file 60 giây, console heartbeat 300 giây. Lưu UTC, PID, CPU, RSS, k, fit seconds, Val Accuracy/F1, checksum checkpoint. CPU% có thể >100 với nhiều luồng; không dùng GPU.

~~~powershell
$taskRun = 'D:\COMPACT-MI-RFE\svm_resumable\mi_rfe_svm\FW_512_FS_16\mi_rfe_svm_FW512_FS16_resumable01'
Get-Content -LiteralPath "$taskRun\progress.json"
Get-Content -LiteralPath "$taskRun\events.jsonl" -Tail 12 -Wait
~~~

Dừng an toàn: ở cửa sổ PowerShell khác, gửi yêu cầu và ĐỢI fit hiện tại kết thúc; command không chụp optimizer giữa fit.

~~~powershell
Set-Location 'C:\Users\LAB_LEE\Desktop\RFforNetwork_GITHUB'
& 'C:\Users\LAB_LEE\Desktop\RFforNetwork_GITHUB\.venv\Scripts\python.exe' -m local.run_mi_rfe_svm request-stop --checkpoint-root 'D:\COMPACT-MI-RFE\svm_resumable\mi_rfe_svm\FW_512_FS_16\mi_rfe_svm_FW512_FS16_resumable01' --session-id 'session02'
~~~

Ctrl+C hoặc hard budget/resource violation có thể dừng fit dở; chỉ checkpoint hoàn tất mới dùng lại. Job Object kill-on-close ngăn worker tiếp tục vô giám sát khi supervisor chết. Nếu Job assignment không được VM/Windows cho phép, BLOCK, không fallback sang chạy trần.

Phiên tiếp: giữ run name, origin binding, source, dữ liệu, profile. Đổi session_id thành session03 trong config; tạo preflight/approval tên mới; chạy lại các bước 8. Không tự chạy phiên tiếp theo, không ghi đè session02. Evidence tests có thể tái sử dụng khi commit/source không đổi. Cùng tên run không có nghĩa là được dùng approval phiên cũ.

Nếu origin lock còn do Colab chết: xác minh Colab đã terminate thật, xem sessions/session01/session_result.json và progress.json. Thiếu report phải được ghi nhận, không suy ra kết thúc sạch. Không xóa active_session.json bằng tay. Takeover đã duyệt sẽ archive lock vào sessions/interrupted_lock_<uuid>.json và giữ lịch sử. Lock đang sống/heartbeat quá mới vẫn chặn.

Sau khi COMPLETED_VERIFIED, kết quả ở checkpoint-root\results\completed\, không phải trong clone hoặc cache. Session result/local_execution ghi riêng môi trường Windows/commit triển khai; scientific binding gốc vẫn là Colab origin. Không diễn giải identity Colab đó là mô tả host Windows.

## 10. Nếu thiếu checkpoint trên VM (không cần nếu đã cài đủ)

Chỉ dùng bản export tin cậy từ chính bạn; joblib/pickle có thể thực thi code khi load, checksum không chứng minh an toàn trước nguồn độc hại.

Dừng Colab trước export. Tải đủ run folder, gồm run_manifest.json, checkpoint.json từng stage, artifact joblib/csv, sessions, results/Test claim, file ẩn và folder .incomplete-*; đừng chỉ tải k_075. Đồng thời giữ approval session01, đúng technical probe và checksum manifest SSD ở environment. Google Drive export có thể bỏ file ẩn; nếu không còn runtime để kiểm kê, không giả định completeness—scanner sẽ BLOCK khi thiếu bằng chứng bắt buộc.

Không cần mở Colab hoặc mount Drive từ Windows runner. Bạn có thể kiểm tra bản Drive đã đồng bộ trên Windows bằng inspect --config-file trỏ config read-only của bản đó; đối chiếu completed_k/next_k rồi mới chọn bản local. Không chạy hai bản run trên hai máy.

Import tùy chọn, không dùng với destination đã có sẵn:

~~~powershell
Set-Location 'C:\Users\LAB_LEE\Desktop\RFforNetwork_GITHUB'
& 'C:\Users\LAB_LEE\Desktop\RFforNetwork_GITHUB\.venv\Scripts\python.exe' -m local.run_mi_rfe_svm import-checkpoints --source 'E:\COLAB_EXPORT\mi_rfe_svm_FW512_FS16_resumable01' --checkpoint-root 'D:\COMPACT-MI-RFE\svm_resumable\mi_rfe_svm\FW_512_FS_16\mi_rfe_svm_FW512_FS16_resumable01'
~~~

Import giữ byte/SHA, không sửa path/hash trong origin manifest; destination bắt buộc vắng mặt, source/target disjoint. Import là kiểm integrity cơ bản, không cấp quyền training; vẫn phải preflight/approve.

## 11. Các giới hạn bắt buộc công bố

- Chưa chạy tests hoặc full continuation. 47/47 PASS cũ không phải bằng chứng cho source local mới.
- Chưa kiểm checkpoint thật của VM; chưa biết k cuối thực sự.
- Nguồn Colab 7718c74 và source khoa học phải khớp; module engine chỉ được khác phép tách adapter đã kiểm AST. Sai khác khác hoặc origin thiếu evidence → MIGRATION_BLOCKED, không sửa hash.
- Fingerprint gốc chỉ metadata + relative paths/sizes, không full-content. Vì vậy bắt buộc checksum manifest export tin cậy và kiểm nội dung riêng; không coi fingerprint đủ.
- Linux→Windows, cùng version packages, vẫn chưa chứng minh tương đương số học bitwise. Giữ ranking/đặc trưng/k cũ, không fit lại MI/RFE để làm đẹp kết quả. Cần công bố thí nghiệm tiếp tục qua hai môi trường.
- Kiểm SHA không bảo vệ dataset nếu tác nhân khác tiếp tục sửa file trong lúc huấn luyện; để dữ liệu immutable/read-only và không chỉnh đồng thời. Cache memmap không port từ Colab; tái nạp Train/Val không phải retrain.
- SVC.fit/final fit dở không resume optimizer. Một fit có thể kéo dài hơn cả budget được duyệt.
- Resource RSS polling không đảm bảo chống mọi OOM; Job Object dùng kill-on-close để giữ custody, không thay scientific memory parameters.
- Job/nested-job support, VM NTFS locking, fsync/rename và full I/O chưa được kiểm thực tế. Không dùng checkpoint trên network share/Drive live sync để train đồng thời; ưu tiên NTFS SSD local.
- Nhãn SCENARIO_PROXY; Benign rất hiếm; capture/session independence chưa được chứng minh.

