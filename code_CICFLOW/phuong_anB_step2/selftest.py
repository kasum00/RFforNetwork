"""Synthetic suite, explicitly run by the user; never a production SVC fit."""
import argparse
import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from .contract import atomic, source_snapshot, redact
from .environment import environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args(argv)
    source = Path(args.source).resolve()
    suite = unittest.defaultTestLoader.discover(str(source / "phuong_anB_step2/tests"), top_level_dir=str(source))
    blocked_svc, blocked_data = [0], [0]
    def forbid_svc(*args, **kwargs):
        blocked_svc[0] += 1
        raise AssertionError("Synthetic suite must not fit a real sklearn SVC")
    import pyarrow.parquet as pq
    original_read, original_footer = pq.read_table, pq.ParquetFile
    production_roots = (Path("/content/drive/MyDrive/CICFLOWMETER_FEATURE_CSV").resolve(),
                        (source.parent / "CICFLOWMETER_FEATURE_CSV").resolve())
    def guarded(function):
        def call(path, *args, **kwargs):
            if isinstance(path, (str, Path)):
                resolved = Path(path).resolve()
                if any(resolved == root or root in resolved.parents for root in production_roots):
                    blocked_data[0] += 1
                    raise AssertionError("Synthetic suite attempted production dataset access")
            return function(path, *args, **kwargs)
        return call
    with patch("sklearn.svm.SVC.fit", forbid_svc), \
         patch("pyarrow.parquet.read_table", guarded(original_read)), \
         patch("pyarrow.parquet.ParquetFile", guarded(original_footer)):
        result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    diagnostics, source_sha, env = [], None, None
    try:
        source_sha = source_snapshot(source)["sha256"]
        env = environment()
        diagnostics.extend(env["errors"])
    except Exception as exc:
        diagnostics.append(redact(f"{type(exc).__name__}: {exc}"))
    record = {"status": "PASS" if result.wasSuccessful() and result.testsRun > 0 and not result.skipped and not diagnostics
              and not blocked_svc[0] and not blocked_data[0] else "FAIL",
              "tests_run": result.testsRun, "skipped": len(result.skipped),
              "failures": len(result.failures), "errors": len(result.errors),
              "source_sha256": source_sha, "environment": env, "diagnostic_errors": diagnostics,
              "real_svc_fit_calls": 0, "real_svc_fit_attempts_blocked": blocked_svc[0],
              "production_dataset_access": bool(blocked_data[0]),
              "production_dataset_attempts_blocked": blocked_data[0]}
    atomic(Path(args.evidence), record, exclusive=True)
    print("__B2_TEST_RESULT__" + json.dumps(record, allow_nan=False), flush=True)
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
