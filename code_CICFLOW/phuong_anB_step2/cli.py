"""User-launched B worker. No final-refit or Test mode exists."""
import argparse
import json
import sys
from .contract import SCIENTIFIC, Blocked, read, redact

MARKER = "__B2_SELECTION_RESULT__"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", help="Local JSON; otherwise JSON stdin")
    args = parser.parse_args(argv)
    try:
        request = read(args.request) if args.request else json.load(sys.stdin)
        from .environment import environment
        env = environment()
        if env["errors"]:
            result = {"status": "BLOCKED", "reasons": env["errors"], "environment": env,
                      "worker_fit_started": False, **SCIENTIFIC}
        else:
            from .pipeline import run
            result = run(request)
        print(MARKER + json.dumps(result, allow_nan=False), flush=True)
        return 0 if result.get("status") in ("PASS", "PAUSED") else 2
    except KeyboardInterrupt:
        result = {"status": "INTERRUPTED", "message": "Only verified COMMITTED units resume; no mid-SVC continuation", **SCIENTIFIC}
        print(MARKER + json.dumps(result), flush=True)
        return 130
    except Exception as exc:
        result = {"status": "BLOCKED" if isinstance(exc, Blocked) else "FAIL",
                  "message": redact(f"{type(exc).__name__}: {exc}"), **SCIENTIFIC}
        print(MARKER + json.dumps(result), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
