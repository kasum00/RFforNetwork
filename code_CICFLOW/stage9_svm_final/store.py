"""Exclusive per-method/config worker leases and verified no-clobber bundles."""
import contextlib
import os
import shutil
import uuid
from pathlib import Path
from .common import Blocked, digest, files_checked, info, intact, once, read, utc


def ensure_identity(root, identity):
    root = Path(root)
    if root.exists():
        if not (root / "identity_manifest.json").is_file() or read(root / "identity_manifest.json") != identity:
            raise Blocked("Existing Stage9 directory identity mismatch; no overwrite")
    else:
        root.mkdir(parents=True, exist_ok=False)
        once(root / "identity_manifest.json", identity)
    intact(identity)


@contextlib.contextmanager
def lease(root, identity, *, previous_stopped=None):
    """Exclusive monotonically numbered durable admission. Never auto-release another worker."""
    root = Path(root)
    ensure_identity(root, identity)
    folder = root / "leases"
    folder.mkdir(exist_ok=True)
    prior = sorted(folder.glob("LEASE_[0-9][0-9][0-9][0-9][0-9][0-9].json"))
    for index, path in enumerate(prior, 1):
        old = read(path)
        if old["identity"] != identity["sha256"] or path.stem != f"LEASE_{index:06d}":
            raise Blocked("Worker lease chain differs")
        release = path.with_name(path.stem + ".RELEASED.json")
        if not release.is_file():
            if index != len(prior) or previous_stopped != "PREVIOUS STAGE9 WORKER STOPPED " + old["token"]:
                raise Blocked("Unreleased worker; confirm actual termination first: " + old["token"])
            # Explicit takeover is recorded separately; old lease is preserved.
            once(release, dict(identity=identity["sha256"], token=old["token"],
                               status="USER_CONFIRMED_TERMINATED", utc=utc()))
        elif read(release).get("token") != old["token"] or read(release).get("identity") != identity["sha256"]:
            raise Blocked("Invalid worker release receipt")
    path = folder / f"LEASE_{len(prior)+1:06d}.json"
    token = uuid.uuid4().hex
    once(path, dict(identity=identity["sha256"], token=token, started=utc(), pid=os.getpid()))
    # Only the winner of exclusive-create reaches user code.
    status = "FAILED_OR_INTERRUPTED"
    try:
        yield token
        status = "RELEASED"
    finally:
        once(path.with_name(path.stem + ".RELEASED.json"),
             dict(identity=identity["sha256"], token=token, status=status, utc=utc()))


def copy_once(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = info(source)
    if target.exists():
        if info(target) != expected:
            raise Blocked("Partial/different artifact preserved; no overwrite: " + str(target))
    else:
        with Path(source).open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, 8*1024**2)
            dst.flush()
            os.fsync(dst.fileno())
    if info(target) != expected:
        raise Blocked("Artifact copy readback mismatch")


def bundle(root, checkpoint, identity):
    root = Path(root)
    if not (root / checkpoint).is_file():
        raise Blocked("Bundle checkpoint not committed; preserve partial artifacts: " + str(root))
    record = read(root / checkpoint)
    intact(record)
    if record["identity"] != identity["sha256"]:
        raise Blocked("Bundle identity mismatch")
    files_checked(root, record["files"])
    return record


def publish_bundle(local, remote, checkpoint, identity):
    record = bundle(local, checkpoint, identity)
    for name in record["files"]:
        copy_once(Path(local) / name, Path(remote) / name)
    copy_once(Path(local) / checkpoint, Path(remote) / checkpoint)
    if bundle(remote, checkpoint, identity) != record:
        raise Blocked("Durable bundle readback differs")
    return record
