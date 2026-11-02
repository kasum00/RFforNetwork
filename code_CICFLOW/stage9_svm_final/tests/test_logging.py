import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from stage9_svm_final.controller import launch
from .fixtures import fixture


class Logging(unittest.TestCase):
    def test_stdout_stderr_append_and_native_status_separation(self):
        with tempfile.TemporaryDirectory() as temporary:
            request, _, _ = fixture(Path(temporary))
            request["mode"] = "preflight"
            for code, status in ((0, "BLOCKED"), (7, "FAIL")):
                fake = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO("progress\n__STAGE9_RESULT__"+json.dumps({"status": status})+"\n"),
                    stderr=io.StringIO("stderr evidence\n"), wait=lambda c=code: c, poll=lambda c=code: c, returncode=code)
                with patch("stage9_svm_final.controller.subprocess.Popen", return_value=fake):
                    result = launch(request)
                self.assertEqual(result["native_exit_code"], code)
                self.assertEqual(result["status"], status)
                self.assertTrue(result["logs_saved"])
            text = Path(result["aggregate_log"]).read_text(encoding="utf-8")
            self.assertEqual(text.count("stderr evidence"), 2)
            self.assertIn('"native_exit_code": 7', text)
            self.assertIn('"program_status": "BLOCKED"', text)

    def test_locked_aggregate_warns_and_keeps_child_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            request, _, _ = fixture(Path(temporary))
            request["mode"] = "preflight"
            aggregate = Path(request["dataset"]).parent / "code_CICFLOW/logs/ALL_EXECUTION_RESULTS.log"
            aggregate.mkdir(parents=True)
            fake = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO('__STAGE9_RESULT__{"status":"BLOCKED"}\n'),
                stderr=io.StringIO("real child error\n"), wait=lambda: 9, poll=lambda: 9, returncode=9)
            with patch("stage9_svm_final.controller.subprocess.Popen", return_value=fake):
                result = launch(request)
            self.assertEqual(result["native_exit_code"], 9)
            self.assertFalse(result["logs_saved"])
            self.assertTrue(result["log_errors"])
