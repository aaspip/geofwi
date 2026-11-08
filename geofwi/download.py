
"""Verified Zenodo downloads for GeoFWI. Standard-library only."""

# =============================================================================
# GeoFWI Dataset Downloader
# =============================================================================
#
# Download and manage the GeoFWI benchmark dataset from Zenodo.
#
# Official dataset:
#   https://zenodo.org/records/24061019
#
# GitHub repository:
#   https://github.com/aaspip/geofwi
#
# =============================================================================
# INSTALLATION
# =============================================================================
#
# Install GeoFWI from the repository root:
#
#   pip install -e .
#
# Install tqdm for a live download progress bar:
#
#   pip install tqdm
#
# =============================================================================
# EXAMPLE 1: BASIC DOWNLOAD (RECOMMENDED)
# =============================================================================
#
#   import geofwi
#
#   path = geofwi.download()
#   print("Dataset location:", path)
#
# Default behavior:
#   - Automatically selects a local dataset directory.
#   - Downloads the file from the official Zenodo record.
#   - Displays a live progress bar if tqdm is installed.
#   - Verifies the downloaded file using its checksum.
#   - Automatically extracts ZIP archives.
#   - Skips downloading if the file already exists and is verified.
#
# Default storage locations:
#   macOS:    ~/Datasets/GeoFWI/
#   Linux:    ~/Datasets/GeoFWI/
#   Windows:  ~/Documents/Datasets/GeoFWI/
#
# =============================================================================
# EXAMPLE 2: DOWNLOAD WITH PROGRESS BAR
# =============================================================================
#
#   import geofwi
#
#   path = geofwi.download(progress=True)
#
# The progress bar displays:
#   - Percentage downloaded
#   - Downloaded size / total size
#   - Download speed
#   - Estimated remaining time
#
# Example:
#   Downloading geofwi.zip: 72%|████████| 17.6M/24.4M [00:06<00:02, 2.8MB/s]
#
# Disable progress reporting:
#
#   path = geofwi.download(progress=False)
#
# =============================================================================
# EXAMPLE 3: DOWNLOAD TO A CUSTOM DIRECTORY
# =============================================================================
#
#   import geofwi
#
#   path = geofwi.download(
#       path="~/Datasets/GeoFWI",
#       progress=True,
#       extract=True,
#   )
#
# Another example using an external drive:
#
#   path = geofwi.download(
#       path="/Volumes/ExternalSSD/GeoFWI",
#       progress=True,
#   )
#
# The destination directory will be created automatically.
#
# =============================================================================
# EXAMPLE 4: SET A PERMANENT DATA DIRECTORY
# =============================================================================
#
# Instead of specifying path every time, set the environment variable:
#
# macOS / Linux:
#
#   export GEOFWI_DATA_DIR="$HOME/Datasets/GeoFWI"
#
# Windows PowerShell:
#
#   $env:GEOFWI_DATA_DIR = "$HOME\Datasets\GeoFWI"
#
# Then use:
#
#   import geofwi
#   path = geofwi.download()
#
# GeoFWI automatically uses the configured directory.
#
# You can also configure the directory within Python:
#
#   import os
#   os.environ["GEOFWI_DATA_DIR"] = "/path/to/GeoFWI"
#
#   import geofwi
#   path = geofwi.download()
#
# =============================================================================
# EXAMPLE 5: LIST AVAILABLE ZENODO FILES
# =============================================================================
#
#   import geofwi
#
#   files = geofwi.available_files()
#
#   for item in files:
#       print(item["name"], item["size"], item["checksum"])
#
# This is useful when the Zenodo record contains multiple files.
#
# =============================================================================
# EXAMPLE 6: DOWNLOAD A SPECIFIC FILE
# =============================================================================
#
#   import geofwi
#
#   files = geofwi.available_files()
#   filename = files[0]["name"]
#
#   path = geofwi.download(
#       filename=filename,
#       progress=True,
#   )
#
# If multiple files exist in the Zenodo record, specify filename.
#
# =============================================================================
# EXAMPLE 7: DOWNLOAD WITHOUT EXTRACTING
# =============================================================================
#
#   import geofwi
#
#   archive = geofwi.download(
#       extract=False,
#       progress=True,
#   )
#
# print("Downloaded archive:", archive)
#
# The ZIP file remains in the dataset directory.
#
# To download and automatically extract:
#
#   path = geofwi.download(extract=True)
#
# =============================================================================
# EXAMPLE 8: FORCE A FRESH DOWNLOAD
# =============================================================================
#
#   import geofwi
#
#   path = geofwi.download(
#       force=True,
#       progress=True,
#   )
#
# force=True downloads the file again even if a verified copy exists.
#
# By default, GeoFWI checks existing files and avoids unnecessary downloads.
#
# =============================================================================
# EXAMPLE 9: REUSE AN EXISTING DATASET
# =============================================================================
#
#   import geofwi
#
#   path = geofwi.download()
#
# If the file has already been downloaded and passes verification,
# GeoFWI skips the network file transfer.
#
# Example output:
#
#   Already downloaded and verified:
#   /Users/username/Datasets/GeoFWI/geofwi.zip
#
# Note: GeoFWI still queries the Zenodo manifest to verify the file.
# ZIP extraction is performed again when extract=True.
#
# =============================================================================
# EXAMPLE 10: FIND THE DEFAULT DATA DIRECTORY
# =============================================================================
#
#   import geofwi
#
#   print(geofwi.data_dir())
#
# Or directly:
#
#   from geofwi.download import data_dir
#
#   print(data_dir())
#
# =============================================================================
# EXAMPLE 11: CALL THE DOWNLOADER DIRECTLY
# =============================================================================
#
# If geofwi.download() is not exposed in geofwi/__init__.py:
#
#   from geofwi.download import download
#
#   path = download(progress=True)
#
# To enable the top-level API, add this to geofwi/__init__.py:
#
#   from .download import download, data_dir, available_files
#
# =============================================================================
# EXAMPLE 12: COMPLETE DOWNLOAD CONFIGURATION
# =============================================================================
#
#   import geofwi
#
#   path = geofwi.download(
#       path="~/Datasets/GeoFWI",  # Local destination directory
#       filename=None,            # Auto-select if exactly one Zenodo file
#       extract=True,             # Automatically extract ZIP archives
#       force=False,              # Reuse existing verified downloads
#       progress=True,            # Display download progress
#   )
#
#   print("GeoFWI dataset is available at:", path)
#
# =============================================================================
# NOTES
# =============================================================================
#
# 1. Internet access is required for the Zenodo metadata request.
# 2. tqdm is optional but recommended for download progress reporting.
# 3. Existing files are checked using Zenodo checksums and file sizes.
# 4. Interrupted downloads currently restart rather than resume.
# 5. ZIP extraction requires additional disk space.
# 6. The returned path points to the extracted directory when extract=True
#    and the downloaded file is a ZIP archive.
# 7. The downloader does not assume a particular NumPy array is present.
# 8. For large datasets, an external SSD can be selected using path=.
#
# =============================================================================


from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

RECORD_ID = '24061019'
API_URL = f'https://zenodo.org/api/records/{RECORD_ID}'


def data_dir() -> Path:
    """Stable, configurable user data directory (not the package installation)."""
    override = os.environ.get('GEOFWI_DATA_DIR')
    if override:
        return Path(override).expanduser().resolve()
    if os.name == 'nt':
        return (Path.home() / 'Documents' / 'Datasets' / 'GeoFWI').resolve()
    return (Path.home() / 'Datasets' / 'GeoFWI').resolve()


def manifest() -> dict:
    req = urllib.request.Request(API_URL, headers={'User-Agent': 'geofwi-benchmark/1.0'})
    with urllib.request.urlopen(req, timeout=40) as response:
        return json.load(response)


def _digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path, checksum: str | None, size: int | None) -> bool:
    if not path.is_file() or (size is not None and path.stat().st_size != size):
        return False
    if checksum:
        algorithm, sep, expected = checksum.partition(':')
        if not sep or algorithm.lower() not in {'md5', 'sha256'}:
            raise ValueError(f'Unsupported checksum format: {checksum}')
        return _digest(path, algorithm.lower()).lower() == expected.lower()
    return True


def _extract_zip(archive: Path, destination: Path) -> None:
    """Reject traversal paths, symlinks and archive bombs before extracting."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        total = sum(i.file_size for i in entries)
        if total > 20 * 1024**3 or len(entries) > 100_000:
            raise ValueError('Archive exceeds safe extraction limits')
        root = destination.resolve()
        for info in entries:
            target = (root / info.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f'Unsafe archive member: {info.filename}')
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Symlink archive member: {info.filename}')
        for info in entries:
            target = (root / info.filename).resolve()
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)


def download(path=None, *, filename=None, extract=True, force=False, progress=True) -> Path:
    """Download one verified Zenodo file, optionally extracting a ZIP archive.

    Does not assume the Zenodo archive contains a particular NumPy array.
    Set progress=False to disable the progress bar (requires optional tqdm).
    """
    target_dir = Path(path).expanduser().resolve() if path else data_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    record = manifest()
    files = record.get('files', [])
    if not files:
        raise RuntimeError('Zenodo record has no downloadable files')
    if filename is None:
        if len(files) != 1:
            raise ValueError('Multiple files exist; specify filename from available_files()')
        item = files[0]
    else:
        item = next((f for f in files if f['key'] == filename), None)
        if item is None:
            raise ValueError(f'Unknown Zenodo filename: {filename}')
    name = item['key']
    if Path(name).name != name:
        raise ValueError('Unexpected file path in Zenodo manifest')
    output = target_dir / name
    checksum = item.get('checksum')
    size = item.get('size')
    if force or not _verify(output, checksum, size):
        free = shutil.disk_usage(target_dir).free
        if size is not None and free < int(size * 1.2):
            raise OSError(f'Insufficient disk space: {free} bytes available')
        url = item.get('links', {}).get('content') or item.get('links', {}).get('self')
        if not url or not url.startswith('https://zenodo.org/'):
            raise ValueError('Zenodo download URL is missing or unexpected')
        partial = output.with_name(output.name + '.part')
        # Re-download incomplete files safely; avoid treating an interrupted file as complete.
        req = urllib.request.Request(url, headers={'User-Agent': 'geofwi-benchmark/1.0'})
        try:
            with urllib.request.urlopen(req, timeout=120) as src, partial.open('wb') as dst:
                total = size or int(src.headers.get('Content-Length', 0)) or None
                if progress and tqdm is None:
                    print('Install tqdm for a download progress bar: python -m pip install tqdm')
                bar = tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024,
                           desc=f'Downloading {name}', disable=not progress) if tqdm else None
                try:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                        if bar is not None:
                            bar.update(len(chunk))
                finally:
                    if bar is not None:
                        bar.close()
            if not _verify(partial, checksum, size):
                raise IOError('Downloaded file failed checksum/size verification')
            partial.replace(output)
        finally:
            partial.unlink(missing_ok=True)
    elif progress:
        print(f'Already downloaded and verified: {output}')
    if extract and output.suffix.lower() == '.zip':
        destination = target_dir / output.stem
        _extract_zip(output, destination)
        shutil.rmtree(destination / "__MACOSX", ignore_errors=True)
        return destination
    return output


def available_files() -> list[dict]:
    """Return the official Zenodo file inventory."""
    return [{'name': f['key'], 'size': f.get('size'), 'checksum': f.get('checksum')}
            for f in manifest().get('files', [])]
