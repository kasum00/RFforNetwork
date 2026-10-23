from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

CONFIG = "FW_512_FS_16"
RUN_NAME = "mi_rfe_svm_FW512_FS16_resumable01"
ORIGIN_COMMIT = "7718c74c44070600ea876a3a1c678632ccb184dd"


def absolute(value) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"Absolute local path required: {value!r}")
    return path.resolve()


def logical_path(value: str) -> str:
    """Normalize relative inventory only; never rewrite origin manifests."""
    if not isinstance(value, str) or not value or ":" in value:
        raise ValueError("Invalid logical path")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("Unsafe relative path")
    return path.as_posix()


def token(value: str) -> str:
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
        raise ValueError("Session ID must use ASCII letters, digits, _ or -")
    return value


@dataclass(frozen=True)
class LocalPaths:
    repo: Path
    data_root: Path
    manifest_root: Path
    checkpoint_root: Path
    output_root: Path

    @classmethod
    def from_args(cls, args):
        result = cls(*(absolute(getattr(args, name)) for name in cls.__dataclass_fields__))
        for path in (result.repo, result.data_root, result.manifest_root, result.checkpoint_root):
            if not path.is_dir():
                raise FileNotFoundError(path)
        if not (result.checkpoint_root / "run_manifest.json").is_file():
            raise PermissionError("MIGRATION_BLOCKED: existing run manifest required; no fresh run")
        for protected in (result.data_root, result.manifest_root, result.checkpoint_root):
            if (result.output_root == protected or result.output_root.is_relative_to(protected)
                    or protected.is_relative_to(result.output_root)):
                raise ValueError("Output/cache must be disjoint from protected inputs")
        return result

    def record(self):
        return {name: str(getattr(self, name)) for name in self.__dataclass_fields__}
