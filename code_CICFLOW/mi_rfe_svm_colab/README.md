# CICFLOW MI–RFE–SVM: local progress, five-hour Drive checkpoints

Entry notebook: `notebooks/MI_RFE_SVM_OPTIMIZED_COLAB.ipynb`.

Read `reports/SVM_COLAB_EXECUTION_GUIDE.md` before running. The default mode is preflight. This package reuses the existing CICFLOW reference numerical functions; it does not replace the old MI, RFE, SVM or notebook sources. It never executes on import.

Pipeline: all 380 Train features → one reference MI call → Top100 → SVM-branch RF-RFE → all k=1..100 → frozen Validation selection → separate final Train+Validation refit → durable Test-once claim → final verification.

K results, console output and cached arrays stay under `/content`. After a completed k, a Drive sync is due only when >=18000 seconds have elapsed since the previous successful sync. MI/RFE completion, explicit checkpoint/stop, final-refit save and final completion are separate durable boundaries. No SVC mid-fit checkpoint exists.

Final Drive folders: `MyDrive/experiment_results/mi_rfe_svm_<FW/FS>_run01`. Existing foreign/legacy or completed folders are protected. Unit tests, preflight, real data/MI/RFE/SVM and the notebook have **not been executed by Codex**. No measured performance improvement or numerical reference equivalence is claimed. Scientific status stays `PROVISIONAL_RESEARCH / OFFICIAL_BLOCKED`.
