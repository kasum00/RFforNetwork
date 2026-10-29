"""Explicit user entry point; no final-refit or Test command exists."""
import argparse
import json
import sys
from .contract import SCIENTIFIC, Blocked, read, redact

MARKER = "__A1_SELECTION_RESULT__"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", help="LOCAL JSON request; otherwise read JSON stdin")
    args = parser.parse_args(argv)
    try:
        request = read(args.request) if args.request else json.load(sys.stdin)
        from .pipeline import run
        result = run(request)
        print(MARKER + json.dumps(result, allow_nan=False), flush=True)
        return 0 if result.get("status") in ("PASS", "PAUSED") else 2
    except KeyboardInterrupt:
        result = {"status": "INTERRUPTED", "message": "Only COMMITTED stages/k can be restored; no SVC mid-fit resume", **SCIENTIFIC}
        print(MARKER + json.dumps(result), flush=True)
        return 130
    except Exception as exc:
        result = {"status": "BLOCKED" if isinstance(exc, Blocked) else "FAIL",
                  "message": redact(f"{type(exc).__name__}: {exc}"), **SCIENTIFIC}
        print(MARKER + json.dumps(result), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
