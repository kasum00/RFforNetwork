"""User-run A1 synthetic tests; no real SVC fit or production dataset access."""
import argparse
import sys
import unittest
from pathlib import Path
from .contract import atomic, source_snapshot
from mi_rfe_svm_colab.preflight import environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args(argv)
    source = Path(args.source).resolve()
    suite = unittest.defaultTestLoader.discover(str(source / "phuong_anA/tests"), top_level_dir=str(source))
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    record = {"status": "PASS" if result.wasSuccessful() and result.testsRun > 0 and not result.skipped else "FAIL",
              "tests_run": result.testsRun, "skipped": len(result.skipped),
              "source_sha256": source_snapshot(source)["sha256"], "environment": environment(),
              "real_svc_fit_calls": 0, "production_dataset_access": False}
    atomic(Path(args.evidence), record, exclusive=True)
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
