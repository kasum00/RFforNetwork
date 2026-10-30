"""Unchanged A1 stream capture implementation, scoped to B local logs."""
import shutil
import threading
from pathlib import Path
from .contract import redact

class Capture:
    def __init__(self, path, stdout, stderr):
        self.path, self.stdout, self.stderr = Path(path), stdout, stderr
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a", encoding="utf-8")
        self.lock, self.offset = threading.RLock(), 0

    def stream(self, is_error=False):
        owner = self
        class Stream:
            def write(self, text):
                clean = redact(text)
                with owner.lock:
                    (owner.stderr if is_error else owner.stdout).write(clean)
                    owner.file.write(clean)
                return len(text)
            def flush(self):
                with owner.lock:
                    owner.file.flush()
                    (owner.stderr if is_error else owner.stdout).flush()
        return Stream()

    def snapshot(self, target):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.file.flush()
            with self.path.open("rb") as src, target.open("xb") as dst:
                src.seek(self.offset)
                shutil.copyfileobj(src, dst, 64*1024)
                self.offset = src.tell()
        return target

    def close(self):
        with self.lock:
            self.file.close()
