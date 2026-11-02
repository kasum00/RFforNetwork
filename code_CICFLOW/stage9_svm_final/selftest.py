"""User-run synthetic evidence; guards real SVC.fit and production Parquet reads."""
import argparse
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from .common import once
from .environment import environment
from .preflight import source_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    import pyarrow.parquet as pq
    originals = dict(read_table=pq.read_table, ParquetFile=pq.ParquetFile)
    def reader(name):
        def guarded(path, *a, **kw):
            if "/content/drive/" in str(path).replace("\\", "/") or "D:/COMPACT-MI-RFE/CICFLOWMETER_FEATURE_CSV" in str(path).replace("\\", "/"):
                raise AssertionError("Synthetic tests attempted a production dataset read")
            return originals[name](path, *a, **kw)
        return guarded
    suite = unittest.defaultTestLoader.discover(str(source / "stage9_svm_final/tests"),
        pattern="test_*.py", top_level_dir=str(source))
    with patch("sklearn.svm.SVC.fit", side_effect=AssertionError("Real SVC.fit is forbidden in synthetic tests")), \
         patch("pyarrow.parquet.read_table", side_effect=reader("read_table")), \
         patch("pyarrow.parquet.ParquetFile", side_effect=reader("ParquetFile")):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = dict(status="PASS" if result.wasSuccessful() and result.testsRun > 0 and not result.skipped else "FAIL",
        tests_run=result.testsRun, skipped=len(result.skipped), errors=len(result.errors), failures=len(result.failures),
        source_sha256=source_snapshot(source)["sha256"], environment=environment(),
        real_svc_training_authorized=False, production_test_authorized=False,
        failure_details=[str(test)+": "+trace for test, trace in result.errors+result.failures])
    once(args.evidence, evidence)
    print("__STAGE9_RESULT__" + json.dumps(evidence), flush=True)
    return 0 if evidence["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
