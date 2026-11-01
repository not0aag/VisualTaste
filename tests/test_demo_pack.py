"""The demo pack: which images it takes, and how it installs safely."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile

import pytest

from visualtaste.db import repo
from visualtaste.jobs.demo import install
from visualtaste.jobs.demo_pack import choose_images


def _corpus(tmp_path, clusters: int = 4, per_cluster: int = 10):
    """Selected images across clusters, a quarter of them from Unsplash."""
    connection = repo.init_db(tmp_path / "db.sqlite")
    n = 0
    with connection:
        for cluster in range(clusters):
            for k in range(per_cluster):
                image_id = f"{n:016x}"
                source = "unsplash_lite" if k % 4 == 0 else "pd12m"
                repo.upsert_image(
                    connection,
                    image_id=image_id,
                    source=source,
                    source_id=str(n),
                    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                    orig_path=f"{image_id}.jpg",
                )
                connection.execute(
                    "UPDATE images SET selected = 1, cluster64 = ? WHERE image_id = ?",
                    (cluster, image_id),
                )
                n += 1
    return connection


class TestChooseImages:
    def test_takes_an_even_share_of_every_cluster(self, tmp_path):
        connection = _corpus(tmp_path)
        chosen = choose_images(connection, 8)
        clusters = [
            connection.execute("SELECT cluster64 FROM images WHERE image_id = ?", (i,)).fetchone()[
                0
            ]
            for i in chosen
        ]
        assert sorted(clusters) == [0, 0, 1, 1, 2, 2, 3, 3]

    def test_never_takes_unsplash(self, tmp_path):
        connection = _corpus(tmp_path)
        chosen = choose_images(connection, 20)
        sources = {
            connection.execute("SELECT source FROM images WHERE image_id = ?", (i,)).fetchone()[0]
            for i in chosen
        }
        assert sources == {"pd12m"}

    def test_is_reproducible(self, tmp_path):
        connection = _corpus(tmp_path)
        assert choose_images(connection, 12) == choose_images(connection, 12)


def _pack(tmp_path, *, with_manifest: bool = True):
    """A tiny pack: data/PACK.json and one file."""
    path = tmp_path / "pack.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        files = {"data/derived/thumb/aa/x.webp": b"webp"}
        if with_manifest:
            files["data/PACK.json"] = json.dumps({"images": 1}).encode()
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return path


class TestInstall:
    def test_installs_a_verified_pack(self, tmp_path):
        pack = _pack(tmp_path)
        digest = hashlib.sha256(pack.read_bytes()).hexdigest()

        manifest = install(pack, tmp_path / "data", digest=digest, force=False)

        assert manifest == {"images": 1}
        assert (tmp_path / "data" / "derived" / "thumb" / "aa" / "x.webp").read_bytes() == b"webp"

    def test_refuses_a_pack_whose_checksum_does_not_match(self, tmp_path):
        with pytest.raises(SystemExit, match="checksum mismatch"):
            install(_pack(tmp_path), tmp_path / "data", digest="0" * 64, force=False)
        assert not (tmp_path / "data").exists()

    def test_will_not_replace_a_collection_unless_forced(self, tmp_path):
        data = tmp_path / "data"
        (data / "db").mkdir(parents=True)
        (data / "db" / "visualtaste.db").write_bytes(b"someone's swipes")

        with pytest.raises(SystemExit, match="already exists"):
            install(_pack(tmp_path), data, digest=None, force=False)
        assert (data / "db" / "visualtaste.db").read_bytes() == b"someone's swipes"

        install(_pack(tmp_path), data, digest=None, force=True)
        assert (data / "PACK.json").exists()

    def test_rejects_an_archive_that_is_not_a_pack(self, tmp_path):
        with pytest.raises(SystemExit, match="not a VisualTaste demo pack"):
            install(
                _pack(tmp_path, with_manifest=False), tmp_path / "data", digest=None, force=False
            )


def test_excluded_images_are_never_chosen(tmp_path):
    connection = _corpus(tmp_path)
    everything = choose_images(connection, 40)
    banned = frozenset(everything[:5])
    assert not banned & set(choose_images(connection, 40, exclude=banned))
