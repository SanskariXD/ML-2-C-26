"""Extract only the official TSV inputs from a Drive-backed ZIP to fast local storage."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
import zlib
from pathlib import Path

from ml2.data import digest_file, write_json

REQUIRED = (
    "train/train_source1.tsv",
    "train/train_source2.tsv",
    "train/train_source3.tsv",
    "train/train_ground_truth.tsv",
    "test/test_source1.tsv",
    "test/test_source2.tsv",
    "test/test_source3.tsv",
)


def members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    selected: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        parts = info.filename.replace("\\", "/").split("/")
        if info.is_dir() or ".." in parts or info.filename.startswith("/"):
            continue
        for name in REQUIRED:
            if len(parts) >= 3 and parts[-3:] == ["dataset", *name.split("/")]:
                if name in selected:
                    raise ValueError(f"Duplicate dataset member: {name}")
                selected[name] = info
    missing = set(REQUIRED) - set(selected)
    if missing:
        raise FileNotFoundError(f"Dataset ZIP missing required TSVs: {sorted(missing)}")
    return selected


def prepare(archive_path: str | Path, destination: str | Path) -> dict:
    archive_path, destination = Path(archive_path), Path(destination)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Drive archive not found: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        selected = members(archive)
        needed = sum(info.file_size for info in selected.values())
        existing = set()
        for name, info in selected.items():
            path = destination / name
            if not path.is_file() or path.stat().st_size != info.file_size:
                continue
            checksum = 0
            with open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                    checksum = zlib.crc32(chunk, checksum)
            if checksum == info.CRC:
                existing.add(name)
        needed -= sum(selected[name].file_size for name in existing)
        destination.mkdir(parents=True, exist_ok=True)
        # Keep room for the index and model. Drive remains the persistent source.
        available = shutil.disk_usage(destination).free
        if available < needed + 2 * 1024**3:
            raise OSError(f"Insufficient local disk: need {needed / 1024**3:.2f} GiB for missing TSVs "
                          f"plus 2 GiB headroom, have {available / 1024**3:.2f} GiB")
        # ZIP CRC is checked by zipfile while each member is streamed. A partial file
        # is never renamed into place, so restarting can safely retry extraction.
        for name, info in selected.items():
            target = destination / name
            if name in existing:
                print(f"Already extracted: {name}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".partial")
            print(f"Extracting {name} ({info.file_size / 1024**2:.1f} MiB)")
            try:
                with archive.open(info) as source, open(temporary, "wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024**2)
                if temporary.stat().st_size != info.file_size:
                    raise ValueError(f"Extraction size mismatch: {name}")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        result = {
            "source_zip": str(archive_path),
            "source_zip_sha256": digest_file(archive_path),
            "members": {
                name: {"zip_member": info.filename, "bytes": info.file_size, "crc32": info.CRC}
                for name, info in selected.items()
            },
            "destination": str(destination),
        }
        write_json(destination / "extraction_manifest.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, dest="archive")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = prepare(args.archive, args.out)
    print(json.dumps({"destination": result["destination"], "files": list(result["members"])}))


if __name__ == "__main__":
    main()
