"""Drive discovery/download uses user-authorized Colab OAuth; no MyDrive assumption."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


SHARED_FOLDER_ID = "1Ixy5iJdeavkEbFJ3cy6LjkQ8h300OAxh"
DRIVE_ROOT = Path("/content/drive/MyDrive/COMPACT-MI-RFE")
FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
_FIELDS = "id,name,mimeType,size,md5Checksum,trashed,shortcutDetails(targetId,targetMimeType)"


def drive_service():
    """Authenticate only in the user's Colab session, never in local tests."""
    from google.colab import auth
    from googleapiclient.discovery import build
    import google.auth
    auth.authenticate_user()
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.readonly"])
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _execute(request, item_id: str):
    try:
        return request.execute()
    except Exception as exc:
        status = getattr(getattr(exc, "resp", None), "status", None)
        if status in (401, 403, 404):
            raise PermissionError(
                f"Cannot read Google Drive item {item_id} (HTTP {status}). "
                "Grant this Colab account Viewer access or check the shared folder ID."
            ) from exc
        raise


def _get(service, file_id: str) -> dict:
    return _execute(service.files().get(fileId=file_id, fields=_FIELDS,
                                        supportsAllDrives=True), file_id)


def _resolve(service, item: dict) -> dict:
    seen = set()
    result = dict(item)
    while result.get("mimeType") == SHORTCUT_MIME:
        target = result.get("shortcutDetails", {}).get("targetId")
        if not target or target in seen:
            raise ValueError(f"Invalid or cyclic Drive shortcut: {item.get('name')}")
        seen.add(target)
        resolved = _get(service, target)
        if resolved.get("trashed"):
            raise FileNotFoundError(f"Drive shortcut target is trashed: {target}")
        resolved["name"] = item["name"]  # Keep the visible name/path.
        result = resolved
    return result


def list_children(service, folder_id: str) -> list[dict]:
    """Return every page, with no first-page or MyDrive-only assumption."""
    if not folder_id or "'" in folder_id or "\\" in folder_id:
        raise ValueError("Unsafe Drive folder ID")
    result, token, seen_tokens = [], None, set()
    while True:
        reply = _execute(service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields=f"nextPageToken,files({_FIELDS})", pageSize=1000,
            pageToken=token, supportsAllDrives=True,
            includeItemsFromAllDrives=True), folder_id)
        result.extend(reply.get("files", []))
        next_token = reply.get("nextPageToken")
        if not next_token:
            return result
        if next_token == token or next_token in seen_tokens:
            raise ValueError("Drive pagination did not advance")
        seen_tokens.add(next_token)
        token = next_token


def _safe_name(name: str) -> str:
    if not name or name in (".", "..") or any(c in name for c in ("/", "\\", ":", "\x00")):
        raise ValueError(f"Unsafe Drive item name: {name!r}")
    return name


def discover_shared_dataset(folder_id: str = SHARED_FOLDER_ID, *, service=None) -> dict:
    """Prefer an extracted WINDOWS_CPLUS folder; ZIP remains an explicit fallback."""
    service = service or drive_service()
    root = _resolve(service, _get(service, folder_id))
    if root.get("trashed") or root.get("mimeType") != FOLDER_MIME:
        raise ValueError("Shared Drive ID is not a readable folder")
    children = [_resolve(service, x) for x in list_children(service, root["id"])]
    windows = [x for x in children if x["name"] == "WINDOWS_CPLUS" and x["mimeType"] == FOLDER_MIME]
    audit = [x for x in children if x["name"] == "CSV_SPLIT_CPLUS" and x["mimeType"] == FOLDER_MIME]
    zips = [x for x in children if x.get("mimeType") == "application/zip"
            or x.get("name", "").lower().endswith(".zip")]
    if len(windows) > 1 or len(audit) > 1:
        raise ValueError("Ambiguous duplicate dataset folders in shared Drive folder")
    return {"mode": "folder" if windows else "zip_available" if zips else "missing",
            "windows_folder": windows[0] if windows else None,
            "audit_folder_present": bool(audit), "zips": zips,
            "children": [{"name": x["name"], "mimeType": x["mimeType"], "id": x["id"]}
                         for x in children], "service": service}


def inventory_window_files(service, folder_id: str) -> list[dict]:
    """Recursively enumerate only WINDOWS_CPLUS; never traverse CSV_SPLIT_CPLUS."""
    files, paths, active = [], set(), set()

    def walk(current_id: str, prefix: Path) -> None:
        if current_id in active:
            raise ValueError("Cyclic Drive folder shortcut")
        active.add(current_id)
        try:
            for child in list_children(service, current_id):
                item = _resolve(service, child)
                name = _safe_name(item["name"])
                relative = prefix / name
                if item["mimeType"] == FOLDER_MIME:
                    walk(item["id"], relative)
                    continue
                if item["mimeType"].startswith("application/vnd.google-apps."):
                    raise ValueError(f"Google-native file is not downloadable CSV: {relative}")
                size = int(item.get("size") or -1)
                md5 = item.get("md5Checksum")
                if size <= 0 or not md5:
                    raise ValueError(f"Drive file lacks positive size/MD5 for safe transfer: {relative}")
                key = relative.as_posix().casefold()
                if key in paths:
                    raise ValueError(f"Duplicate Drive output path: {relative}")
                paths.add(key)
                files.append({"id": item["id"], "relative_path": relative.as_posix(),
                              "size": size, "md5Checksum": md5})
        finally:
            active.remove(current_id)

    walk(folder_id, Path())
    return sorted(files, key=lambda x: x["relative_path"])


def download_drive_file(service, item: dict, destination: Path, progress=None) -> None:
    """Download one file to a fresh path; caller verifies MD5 and handles promotion."""
    from googleapiclient.http import MediaIoBaseDownload
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        downloader = MediaIoBaseDownload(
            handle, service.files().get_media(fileId=item["id"], supportsAllDrives=True),
            chunksize=16 * 1024 * 1024)
        done = False
        while not done:
            try:
                status, done = downloader.next_chunk()
            except Exception as exc:
                code = getattr(getattr(exc, "resp", None), "status", None)
                if code in (401, 403, 404):
                    raise PermissionError(
                        f"Cannot download Drive file {item['id']} (HTTP {code}); "
                        "grant this Colab account Viewer access to the file/folder."
                    ) from exc
                raise
            if progress and status:
                progress(status.progress())


def inspect_shared_folder(folder_id: str = SHARED_FOLDER_ID) -> list[dict]:
    """Backward-compatible full paginated listing (without downloading)."""
    service = drive_service()
    folder = _get(service, folder_id)
    if folder.get("trashed") or folder.get("mimeType") != FOLDER_MIME:
        raise ValueError("Shared folder is unavailable or is not a folder")
    return [{"folder": folder["name"], **item} for item in list_children(service, folder_id)]


def download_archive(file_id: str, destination: Path) -> dict:
    """Stream selected ZIP; never overwrite existing local file."""
    from google.colab import auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import google.auth
    if destination.exists() or destination.with_suffix(destination.suffix + ".partial").exists():
        raise FileExistsError(destination)
    auth.authenticate_user()
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.readonly"])
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    meta = service.files().get(fileId=file_id, fields="id,name,mimeType,size,md5Checksum", supportsAllDrives=True).execute()
    if meta.get("mimeType") != "application/zip" and not str(meta.get("name", "")).lower().endswith(".zip"):
        raise ValueError("Selected Drive file is not a ZIP archive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        with partial.open("xb") as handle:
            downloader = MediaIoBaseDownload(handle, service.files().get_media(fileId=file_id, supportsAllDrives=True), chunksize=16 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        if partial.stat().st_size != int(meta["size"]):
            raise IOError("Downloaded ZIP size differs from Drive metadata")
        md5 = hashlib.md5()
        with partial.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                md5.update(block)
        if meta.get("md5Checksum") and md5.hexdigest() != meta["md5Checksum"]:
            raise IOError("Downloaded ZIP MD5 differs from Drive")
        os.replace(partial, destination)
    except BaseException:
        # Preserve partial evidence; a later attempt must use a new destination.
        raise
    return {"path": str(destination), "drive_name": meta["name"], "bytes": destination.stat().st_size,
            "drive_md5": meta.get("md5Checksum"), "md5_verified": bool(meta.get("md5Checksum"))}


def require_writable_result_root(root: Path = DRIVE_ROOT) -> Path:
    mount = Path("/content/drive/MyDrive")
    if not mount.is_dir():
        raise FileNotFoundError("Google Drive is not mounted at /content/drive/MyDrive")
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".stage6_write_probe"
    if probe.exists():
        raise FileExistsError(probe)
    with probe.open("x", encoding="utf-8") as handle:
        handle.write("write-check\n")
    probe.unlink()
    for name in ("experiment_results", "reports", "checkpoints", "environment"):
        (root / name).mkdir(exist_ok=True)
    return root
