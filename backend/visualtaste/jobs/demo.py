"""Install the demo pack, so the app runs without building a collection.

    python -m visualtaste.jobs.demo --download          # from the GitHub release
    python -m visualtaste.jobs.demo --from demo-pack.tar.gz

The pack is unpacked into the data folder (`data/`, or `VT_DATA_ROOT`). The
download is checked against its published SHA-256 before anything is
written, and the job refuses to overwrite an existing collection unless told
to, because the data folder may hold someone's real swipes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import httpx

from visualtaste.config import get_settings

RELEASE = "https://github.com/not0aag/VisualTaste/releases/download/demo-pack"
PACK_NAME = "visualtaste-demo-pack.tar.gz"


def download(url: str, target: Path) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_bytes(1 << 20):
                handle.write(chunk)


def expected_digest(sha_file: Path) -> str:
    return sha_file.read_text(encoding="utf-8").split()[0].strip().lower()


def install(pack: Path, data_root: Path, *, digest: str | None, force: bool) -> dict:
    """Verify, then unpack `pack` so its `data/` lands at `data_root`."""
    if digest is not None:
        actual = hashlib.sha256(pack.read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"checksum mismatch: expected {digest}, got {actual}; not installed")

    existing_db = data_root / "db" / "visualtaste.db"
    if existing_db.exists() and not force:
        raise SystemExit(
            f"{existing_db} already exists. The demo pack would replace this collection "
            "and its database. Use --force to replace it, or point VT_DATA_ROOT at an "
            "empty folder."
        )

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(pack, "r:gz") as archive:
            # `data` filter: no absolute paths, no links out, no device files.
            archive.extractall(tmp, filter="data")
        unpacked = Path(tmp) / "data"
        if not (unpacked / "PACK.json").exists():
            raise SystemExit(f"{pack} is not a VisualTaste demo pack (no PACK.json)")
        data_root.mkdir(parents=True, exist_ok=True)
        for item in unpacked.iterdir():
            target = data_root / item.name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            shutil.move(str(item), str(target))
    return json.loads((data_root / "PACK.json").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install the VisualTaste demo pack.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--download", action="store_true", help="fetch it from the release")
    source.add_argument("--from", dest="path", type=Path, help="a pack file already on disk")
    parser.add_argument("--url", default=f"{RELEASE}/{PACK_NAME}")
    parser.add_argument("--force", action="store_true", help="replace an existing collection")
    args = parser.parse_args(argv)

    settings = get_settings()
    with tempfile.TemporaryDirectory() as tmp:
        if args.download:
            pack = Path(tmp) / PACK_NAME
            sha = Path(tmp) / f"{PACK_NAME}.sha256"
            print(f"downloading {args.url} ...")
            download(args.url, pack)
            download(f"{args.url}.sha256", sha)
            digest = expected_digest(sha)
        else:
            pack = args.path
            sha = pack.with_name(pack.name + ".sha256")
            digest = expected_digest(sha) if sha.exists() else None
            if digest is None:
                print(f"note: no {sha.name} beside the pack; skipping the checksum")
        manifest = install(pack, settings.data, digest=digest, force=args.force)

    print(
        f"installed {manifest['images']:,} images into {settings.data}\n"
        "next: start the app with `just dev` (or scripts/dev.ps1 / scripts/dev.sh)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
