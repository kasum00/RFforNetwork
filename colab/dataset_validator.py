"""Portable C+ window inventory and safe one-time local preparation."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath


EXPECTED = {
    "FW_256_FS_8": {"train": 1_348_317, "val": 449_343, "test": 449_251},
    "FW_256_FS_16": {"train": 674_169, "val": 224_677, "test": 224_631},
    "FW_512_FS_8": {"train": 1_347_101, "val": 448_735, "test": 448_563},
    "FW_512_FS_16": {"train": 673_559, "val": 224_373, "test": 224_287},
}
WINDOW_ROOT = Path("/content/datasets/CICIoMT24/WINDOWS_CPLUS")
_MIN_TRAINING_DISK_RESERVE = 8 * 1024**3  # memmaps, temp arrays and working margin; not a RAM estimate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_archive_members(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, Path]]:
    entries, seen = [], set()
    for member in archive.infolist():
        name = member.filename.replace("\\", "/")
        pure = PurePosixPath(name)
        if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts) or ":" in name:
            raise ValueError(f"Unsafe ZIP path: {name!r}")
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError(f"ZIP symlink is forbidden: {name!r}")
        parts = pure.parts
        if "WINDOWS_CPLUS" in parts:
            relative = Path(*parts[parts.index("WINDOWS_CPLUS") + 1:])
        elif parts[0] in EXPECTED or parts[0] == "window_run_summary.json":
            relative = Path(*parts)
        else:
            continue  # Ignore wrappers / unrelated artifacts, never extract them.
        if member.is_dir() or str(relative) == ".":
            continue
        key = relative.as_posix().casefold()
        if key in seen:
            raise ValueError(f"Duplicate normalized ZIP output: {relative}")
        seen.add(key)
        entries.append((member, relative))
    if "window_run_summary.json" not in seen:
        raise ValueError("ZIP lacks window_run_summary.json")
    for config in EXPECTED:
        if not any(p.parts[0] == config and p.suffix.lower() == ".csv" for _, p in entries):
            raise ValueError(f"ZIP lacks CSV files for {config}")
    return entries


def inspect_archive(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        entries = _safe_archive_members(archive)
        return {"path": str(path), "compressed_bytes": path.stat().st_size,
                "selected_files": len(entries),
                "selected_uncompressed_bytes": sum(m.file_size for m, _ in entries),
                "csv_files": sum(p.suffix.lower() == ".csv" for _, p in entries),
                "sample_paths": [str(p) for _, p in entries[:8]],
                "crc_full_verified": False}


def _copy_readonly_metadata(repo: Path, target: Path) -> None:
    source = repo / "configs" / "cplus_manifest"
    target.mkdir(parents=True)
    for name in ("split_manifest.csv", "validation.json"):
        shutil.copy2(source / name, target / name)


def validate_dataset(window_root: Path, repo: Path, *, all_headers: bool = True) -> dict:
    """Checks approved manifest, 322-file inventory/schema, not Test feature rows."""
    import sys
    source_path = str(repo / "huyen_code")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    import ciciomt24_sliding_window as sw
    import ciciomt24_training as training
    manifest_root = window_root.parent / "CSV_SPLIT_CPLUS"
    segments = sw.load_manifest(manifest_root)
    sw.verify_split_artifacts(manifest_root, segments)
    computed = sw.expected_window_table(segments)
    for config, roles in EXPECTED.items():
        fw, fs = map(int, (config.split("_")[1], config.split("_")[3]))
        if computed[f"FW{fw}_FS{fs}"] != roles:
            raise ValueError(f"Approved count mismatch for {config}")
    store = training.WindowStore(window_root, split_root=manifest_root)
    if all_headers:
        for files in store.files.values():
            for item in files:
                if item.role != "test":
                    store.validate_header(item.path)
    summary = window_root / "window_run_summary.json"
    inventory = sorted((str(f.path.relative_to(window_root)), f.path.stat().st_size)
                       for files in store.files.values() for f in files)
    if len(inventory) != 322 or any(size <= 0 for _, size in inventory):
        raise ValueError("Window file inventory differs from 322 nonempty source segments")
    fingerprint_payload = json.dumps({"manifest_sha256": sha256(manifest_root / "split_manifest.csv"),
                                      "validation_sha256": sha256(manifest_root / "validation.json"),
                                      "summary_sha256": sha256(summary), "inventory_sizes": inventory},
                                     sort_keys=True).encode()
    return {"status": "PASS_METADATA_SCHEMA", "counts": EXPECTED,
            "label_origin": "SCENARIO_PROXY", "features": len(training.FEATURE_NAMES),
            "output_columns": 394, "metadata_and_label_columns": 14,
            "csv_files": len(inventory), "manifest_segments": len(segments),
            "proxy_window_counts": sw.expected_windows_by_proxy_label(segments),
            "dataset_fingerprint_sha256": hashlib.sha256(fingerprint_payload).hexdigest(),
            "fingerprint_scope": "manifest+summary+CSV relative paths/sizes; not full CSV content hash",
            "test_feature_rows_read": 0, "test_headers_deferred_until_after_refit": True,
            "full_csv_contents_reverified": False,
            "stage42_full_readback_reused": True}


def prepare_dataset(repo: Path, *, archive: Path | None = None,
                    extracted_window_dir: Path | None = None,
                    output_root: Path = WINDOW_ROOT) -> dict:
    """Reuse valid local data or create an isolated .incomplete tree and promote."""
    if output_root.exists():
        return {"action": "reused_verified", **validate_dataset(output_root, repo)}
    if (archive is None) == (extracted_window_dir is None):
        raise ValueError("Provide exactly one ZIP or existing extracted WINDOWS_CPLUS directory")
    dataset_parent = output_root.parent
    incomplete = dataset_parent.with_name(dataset_parent.name + ".incomplete")
    if dataset_parent.exists() or incomplete.exists():
        raise FileExistsError("Dataset target/partial exists; inspect it manually, never merge or overwrite")
    dataset_parent.parent.mkdir(parents=True, exist_ok=True)
    selected_bytes = 0
    if archive is not None:
        info = inspect_archive(archive)
        selected_bytes = info["selected_uncompressed_bytes"]
    else:
        if not (extracted_window_dir / "window_run_summary.json").is_file():
            raise ValueError("Selected extracted folder is not WINDOWS_CPLUS")
        selected_bytes = sum(p.stat().st_size for p in extracted_window_dir.rglob("*") if p.is_file())
    free = shutil.disk_usage(dataset_parent.parent).free
    required = selected_bytes + _MIN_TRAINING_DISK_RESERVE
    if free < required:
        raise OSError(f"Local disk admission FAIL: need {required:,} free bytes, have {free:,}")
    windows = incomplete / "WINDOWS_CPLUS"
    try:
        if archive is not None:
            windows.mkdir(parents=True)
            with zipfile.ZipFile(archive) as handle:
                entries = _safe_archive_members(handle)
                bad = handle.testzip()  # Full CRC check; original ZIP is never changed.
                if bad is not None:
                    raise ValueError(f"ZIP CRC failed at {bad}")
                for member, relative in entries:
                    target = windows / relative
                    if not target.resolve().is_relative_to(windows.resolve()):
                        raise ValueError("ZIP path escapes destination")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with handle.open(member) as source, target.open("xb") as dest:
                        shutil.copyfileobj(source, dest, 8 * 1024 * 1024)
        else:
            shutil.copytree(extracted_window_dir, windows)
        _copy_readonly_metadata(repo, incomplete / "CSV_SPLIT_CPLUS")
        validated = validate_dataset(windows, repo)
        os.replace(incomplete, dataset_parent)
        return {"action": "prepared_verified", "disk_free_before_bytes": free,
                "disk_required_bytes": required, **validated}
    except BaseException:
        # Preserve .incomplete evidence; do not auto-delete or treat as valid.
        raise
