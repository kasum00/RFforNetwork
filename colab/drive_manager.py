"""Drive discovery/download uses user-authorized Colab OAuth; no MyDrive assumption."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


SHARED_FOLDER_ID = "1Ixy5iJdeavkEbFJ3cy6LjkQ8h300OAxh"
DRIVE_ROOT = Path("/content/drive/MyDrive/COMPACT-MI-RFE")


def inspect_shared_folder(folder_id: str = SHARED_FOLDER_ID) -> list[dict]:
    """Prompts notebook user for Google consent; this agent does not authenticate."""
    from google.colab import auth
    from googleapiclient.discovery import build
    import google.auth
    auth.authenticate_user()
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.readonly"])
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    folder = service.files().get(fileId=folder_id, fields="id,name,mimeType,trashed", supportsAllDrives=True).execute()
    if folder.get("trashed") or folder.get("mimeType") != "application/vnd.google-apps.folder":
        raise ValueError("Shared folder is unavailable or is not a folder")
    children, token = [], None
    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken,files(id,name,mimeType,size,md5Checksum)",
            pageSize=1000, pageToken=token, supportsAllDrives=True,
            includeItemsFromAllDrives=True).execute()
        children.extend(response.get("files", []))
        token = response.get("nextPageToken")
        if not token:
            break
    return [{"folder": folder["name"], **item} for item in children]


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
