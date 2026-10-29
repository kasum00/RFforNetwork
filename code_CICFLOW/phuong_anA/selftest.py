"""User-run A1 synthetic tests; no real SVC fit or production dataset access."""
import argparse
import sys
import unittest
from pathlib import Path
from .contract import atomic, source_snapshot, redact
from mi_rfe_svm_colab.preflight import environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args(argv)
    source = Path(args.source).resolve()
    suite = unittest.defaultTestLoader.discover(str(source / "phuong_anA/tests"), top_level_dir=str(source))
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    diagnostics, source_sha, env = [], None, None
    try:
        source_sha = source_snapshot(source)["sha256"]
    except Exception as exc:
        diagnostics.append("Source snapshot unavailable: " + redact(f"{type(exc).__name__}: {exc}"))
    try:
        env = environment()
    except Exception as exc:
        diagnostics.append("Environment unavailable: " + redact(f"{type(exc).__name__}: {exc}"))
    record = {"status": "PASS" if result.wasSuccessful() and result.testsRun > 0 and not result.skipped and not diagnostics else "FAIL",
              "tests_run": result.testsRun, "skipped": len(result.skipped),
              "failures": len(result.failures), "errors": len(result.errors),
              "source_sha256": source_sha, "environment": env, "diagnostic_errors": diagnostics,
              "real_svc_fit_calls": 0, "production_dataset_access": False}
    atomic(Path(args.evidence), record, exclusive=True)
    for message in diagnostics:
        print("EVIDENCE FAIL: " + message, file=sys.stderr, flush=True)
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
