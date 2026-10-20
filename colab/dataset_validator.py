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


def manifest_root_for(window_root: Path, repo: Path) -> Path:
    """Flow CSVs are not required for training; use the portable C+ manifest."""
    for candidate in (window_root.parent / "CSV_SPLIT_CPLUS", repo / "configs" / "cplus_manifest"):
        if all((candidate / name).is_file() for name in ("split_manifest.csv", "validation.json")):
            return candidate
    raise FileNotFoundError("C+ split_manifest.csv/validation.json absent locally and in repository")


def validate_dataset(window_root: Path, repo: Path, *, all_headers: bool = True) -> dict:
    """Checks approved manifest, 322-file inventory/schema, not Test feature rows."""
    import sys
    source_path = str(repo / "huyen_code")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    import ciciomt24_sliding_window as sw
    import ciciomt24_training as training
    manifest_root = manifest_root_for(window_root, repo)
    segments = sw.load_manifest(manifest_root)
    sw.verify_split_artifacts(manifest_root, segments)
    computed = sw.expected_window_table(segments)
    for config, roles in EXPECTED.items():
        fw, fs = map(int, (config.split("_")[1], config.split("_")[3]))
        if computed[f"FW{fw}_FS{fs}"] != roles:
            raise ValueError(f"Approved count mismatch for {config}")
    expected_paths = [sw._expected_window_output_path(window_root, segment, fw, fs)
                      for segment in segments for fw, fs in sw.WINDOW_CONFIGS
                      if sw.window_count(segment.flow_count, fw, fs)]
    missing = [str(path.relative_to(window_root)) for path in expected_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} window CSV(s): {missing[:20]}")
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
            "config_directories": sum((window_root / config).is_dir() for config in EXPECTED),
            "dataset_path": str(window_root), "manifest_path": str(manifest_root),
            "missing_files": [],
            "proxy_window_counts": sw.expected_windows_by_proxy_label(segments),
            "dataset_fingerprint_sha256": hashlib.sha256(fingerprint_payload).hexdigest(),
            "fingerprint_scope": "manifest+summary+CSV relative paths/sizes; not full CSV content hash",
            "test_feature_rows_read": 0, "test_headers_deferred_until_after_refit": True,
            "full_csv_contents_reverified": False,
            "stage42_full_readback_reused": True}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory_precheck(entries: list[dict]) -> None:
    paths = [item["relative_path"] for item in entries]
    if "window_run_summary.json" not in paths:
        raise FileNotFoundError("WINDOWS_CPLUS/window_run_summary.json is missing")
    csv_paths = [path for path in paths if path.endswith("_windows.csv")]
    if len(csv_paths) != 322:
        raise FileNotFoundError(f"Expected 322 window CSVs; found {len(csv_paths)} (missing or extra files)")
    missing_configs = [config for config in EXPECTED
                       if not any(path.startswith(config + "/") for path in csv_paths)]
    if missing_configs:
        raise FileNotFoundError(f"Missing FW/FS configuration folder(s): {missing_configs}")


def _reuse_completed(output_root: Path, repo: Path) -> dict | None:
    if not output_root.exists():
        return None
    stage = output_root.parent.with_name(output_root.parent.name + ".incomplete")
    if stage.exists():
        raise FileExistsError(f"Both completed and incomplete dataset trees exist: {stage}")
    return {"action": "reused_verified", **validate_dataset(output_root, repo)}


def _prepare_incremental(repo: Path, output_root: Path, entries: list[dict],
                         source_kind: str, transfer, *, source_id: str) -> dict:
    """One verified file at a time; an existing .incomplete tree is inspected, not deleted."""
    reused = _reuse_completed(output_root, repo)
    if reused is not None:
        return reused
    _inventory_precheck(entries)
    parent = output_root.parent
    stage = parent.with_name(parent.name + ".incomplete")
    if parent.exists():
        raise FileExistsError(f"Dataset target exists but is not validated: {parent}")
    selected = [item for item in entries if item["relative_path"] == "window_run_summary.json"
                or item["relative_path"].endswith("_windows.csv")]
    plan = {"source_kind": source_kind, "source_id": source_id, "files": selected}
    plan_bytes = json.dumps(plan, sort_keys=True, ensure_ascii=False).encode("utf-8")
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    plan_path = stage / "transfer_plan.json"
    if stage.exists():
        if not plan_path.is_file():
            raise FileExistsError(f"Unrecognized .incomplete tree; inspect manually: {stage}")
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        if previous.get("plan_sha256") != plan_hash:
            raise ValueError("Drive/source inventory changed since .incomplete transfer; inspect manually")
    else:
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.mkdir()
        plan_path.write_text(json.dumps({"plan_sha256": plan_hash, **plan}, indent=2), encoding="utf-8")
    windows = stage / "WINDOWS_CPLUS"
    windows.mkdir(exist_ok=True)
    pending = []
    for item in selected:
        relative = Path(item["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe dataset relative path: {relative}")
        destination = windows / relative
        partial = destination.with_name(destination.name + ".partial")
        if partial.exists():
            expected_hash = item.get("md5Checksum") or _md5(Path(source_id) / relative)
            if partial.stat().st_size == item["size"] and _md5(partial) == expected_hash:
                os.replace(partial, destination)  # Complete bytes survived interruption before rename.
            else:
                raise FileExistsError(f"Incomplete/invalid partial needs inspection; not overwritten: {partial}")
        if destination.exists():
            if destination.stat().st_size != item["size"]:
                raise IOError(f"Completed staged file size changed: {destination}")
            expected_hash = item.get("md5Checksum")
            if expected_hash and _md5(destination) != expected_hash:
                raise IOError(f"Completed staged file MD5 changed: {destination}")
            if not expected_hash and _md5(destination) != _md5(Path(source_id) / relative):
                raise IOError(f"Completed staged file differs from mounted source: {destination}")
        else:
            pending.append(item)
    remaining = sum(item["size"] for item in pending)
    free = shutil.disk_usage(stage.parent).free
    required = remaining + _MIN_TRAINING_DISK_RESERVE
    if free < required:
        raise OSError(f"Local disk admission FAIL: need {required:,} bytes free, have {free:,}; "
                      f"short by {required-free:,} bytes")
    total = len(selected)
    for index, item in enumerate(selected, 1):
        destination = windows / item["relative_path"]
        if destination.exists():
            print(f"[dataset {index}/{total}] verified existing: {item['relative_path']}", flush=True)
            continue
        partial = destination.with_name(destination.name + ".partial")
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[dataset {index}/{total}] transfer {item['relative_path']} ({item['size']:,} bytes)", flush=True)
        transfer(item, partial)
        if partial.stat().st_size != item["size"]:
            raise IOError(f"Downloaded/copied size mismatch: {partial}")
        expected_hash = item.get("md5Checksum") or _md5(Path(source_id) / item["relative_path"])
        if _md5(partial) != expected_hash:
            raise IOError(f"Downloaded/copied MD5 mismatch: {partial}")
        os.replace(partial, destination)
    validated = validate_dataset(windows, repo)
    os.replace(stage, parent)
    return {"action": "prepared_verified", "source_kind": source_kind,
            "disk_free_before_bytes": free, "disk_required_bytes": required,
            "transferred_files": len(pending), **validated}


def prepare_drive_folder_dataset(repo: Path, windows_folder_id: str, service,
                                 *, output_root: Path = WINDOW_ROOT) -> dict:
    """Use Drive API only for WINDOWS_CPLUS; never download CSV_SPLIT_CPLUS."""
    from .drive_manager import download_drive_file, inventory_window_files
    reused = _reuse_completed(output_root, repo)
    if reused is not None:
        return reused
    entries = inventory_window_files(service, windows_folder_id)
    return _prepare_incremental(repo, output_root, entries, "drive_folder",
                                lambda item, path: download_drive_file(service, item, path),
                                source_id=windows_folder_id)


def prepare_mounted_folder_dataset(repo: Path, source: Path,
                                   *, output_root: Path = WINDOW_ROOT) -> dict:
    """Copy a mounted WINDOWS_CPLUS one verified file at a time, or reuse local output."""
    reused = _reuse_completed(output_root, repo)
    if reused is not None:
        return reused
    if source.name != "WINDOWS_CPLUS" or not source.is_dir():
        raise ValueError("Mounted source must point directly to WINDOWS_CPLUS")
    entries = []
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Mounted dataset symlink is not accepted: {path}")
        if path.is_file():
            relative = path.relative_to(source).as_posix()
            entries.append({"relative_path": relative, "size": path.stat().st_size,
                            "mtime_ns": path.stat().st_mtime_ns})
    entries.sort(key=lambda item: item["relative_path"])
    return _prepare_incremental(repo, output_root, entries, "mounted_folder",
                                lambda item, path: shutil.copyfile(source / item["relative_path"], path),
                                source_id=str(source.resolve()))


def prepare_dataset(repo: Path, *, archive: Path | None = None,
                    extracted_window_dir: Path | None = None,
                    output_root: Path = WINDOW_ROOT) -> dict:
    """Reuse valid local data or create an isolated .incomplete tree and promote."""
    reused = _reuse_completed(output_root, repo)
    if reused is not None:
        return reused
    if (archive is None) == (extracted_window_dir is None):
        raise ValueError("Provide exactly one ZIP or existing extracted WINDOWS_CPLUS directory")
    if extracted_window_dir is not None:
        return prepare_mounted_folder_dataset(repo, extracted_window_dir, output_root=output_root)
    dataset_parent = output_root.parent
    incomplete = dataset_parent.with_name(dataset_parent.name + ".incomplete")
    if dataset_parent.exists() or incomplete.exists():
        raise FileExistsError("Dataset target/partial exists; inspect it manually, never merge or overwrite")
    dataset_parent.parent.mkdir(parents=True, exist_ok=True)
    selected_bytes = 0
    if archive is not None:
        info = inspect_archive(archive)
        selected_bytes = info["selected_uncompressed_bytes"]
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
        _copy_readonly_metadata(repo, incomplete / "CSV_SPLIT_CPLUS")
        validated = validate_dataset(windows, repo)
        os.replace(incomplete, dataset_parent)
        return {"action": "prepared_verified", "disk_free_before_bytes": free,
                "disk_required_bytes": required, **validated}
    except BaseException:
        # Preserve .incomplete evidence; do not auto-delete or treat as valid.
        raise
