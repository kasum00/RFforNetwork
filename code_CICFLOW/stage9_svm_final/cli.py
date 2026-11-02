"""User-invoked worker; preserves structured scientific status separately from exit."""
import argparse
import json
import sys
from .common import read

MARKER = "__STAGE9_RESULT__"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request")
    args = parser.parse_args()
    request = read(args.request) if args.request else json.load(sys.stdin)
    try:
        from .environment import environment
        env = environment()
        if env["errors"]:
            raise ValueError("Environment blocked: " + str(env["errors"]))
        from .pipeline import execute
        result = execute(request)
        print(MARKER + json.dumps(result, allow_nan=False), flush=True)
        return 0 if result.get("status") == "PASS" else 2
    except KeyboardInterrupt:
        print(MARKER + json.dumps(dict(status="BLOCKED", reason="Interrupted; inspect durable state; never retry a Test claim")), flush=True)
        return 130
    except Exception as exc:
        from mi_rfe_svm_colab.common import redact
        message = redact(f"{type(exc).__name__}: {exc}")
        print(message, file=sys.stderr, flush=True)
        print(MARKER + json.dumps(dict(status="BLOCKED", reason=message)), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
