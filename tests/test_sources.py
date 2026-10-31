"""Tests for the four source modules: planning, gates, attribution, resume.

Nothing here touches the network. The Megalith tests build a synthetic
WebDataset tar in `tmp_path` with the same layout as the real mirror shards
(`<key>.jpg` plus `<key>.json` / `<key>.txt` sidecars), which is what lets the
extraction and resume logic be tested without a 1.8 GB download.
"""

from __future__ import annotations

import io
import json
import tarfile

import pandas as pd
import pytest

from visualtaste.config import Settings
from visualtaste.data.http import FetchStats, image_id_for
from visualtaste.data.manifest import Manifest
from visualtaste.data.sources import dimensions_acceptable, megalith, openimages, pd12m
from visualtaste.data.sources import unsplash_lite as unsplash

JPEG_BYTES = bytes.fromhex("ffd8ffe000104a46494600010100000100010000") + b"\x00" * 200


@pytest.fixture
def settings(tmp_path):
    return Settings(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        archive_root=tmp_path / "archive",
        min_free_gb=0,
        max_original_mb=8,
        seed=1729,
    )


class TestDimensionGate:
    def test_accepts_an_ordinary_photograph(self):
        assert dimensions_acceptable(1024, 768) is True

    def test_rejects_a_short_side_under_256(self):
        assert dimensions_acceptable(2000, 200) is False

    def test_rejects_a_panorama_wider_than_four_to_one(self):
        assert dimensions_acceptable(5000, 1000) is False
        assert dimensions_acceptable(4000, 1000) is True  # exactly 4:1 is allowed

    def test_unknown_dimensions_pass(self):
        # Absent metadata is not evidence of a bad image; ingestion will decide
        # against the real pixels.
        assert dimensions_acceptable(None, None) is True
        assert dimensions_acceptable(1024, None) is True

    def test_nonsense_dimensions_pass_through_to_ingestion(self):
        assert dimensions_acceptable(0, 0) is True


# --------------------------------------------------------------------------
# Megalith
# --------------------------------------------------------------------------


def build_shard_tar(path, keys_and_bytes):
    """Write a tar laid out the way an img2dataset WebDataset shard is.

    Each record is three members: the image, a JSON sidecar and a caption text
    file. The extractor must take only the image.
    """
    with tarfile.open(path, "w") as tar:
        for key, data in keys_and_bytes:
            for suffix, payload in (
                (".jpg", data),
                (".json", json.dumps({"key": key}).encode()),
                (".txt", b"a caption"),
            ):
                info = tarfile.TarInfo(name=f"{key}{suffix}")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))


def shard_metadata(keys):
    """The `key -> row` mapping `_load_shard_metadata` would return."""
    return {
        key: {
            "key": key,
            "url": f"https://farm1.staticflickr.example/{key}_b.jpg",
            # The mirror stores the Flickr photo page in `caption`.
            "caption": f"https://www.flickr.com/photos/user/{key}",
            "status": "success",
            "width": 1024.0,
            "height": 768.0,
        }
        for key in keys
    }


class TestMegalithShardPlanning:
    def test_asks_for_one_shard_more_than_arithmetic_requires(self):
        # 30,000 images / 10,000 per shard = 3, plus a shard of slack.
        assert megalith.shards_needed(30_000) == 4

    def test_a_small_target_still_needs_a_shard(self):
        assert megalith.shards_needed(1) == 2
        assert megalith.shards_needed(10_000) == 2

    def test_never_returns_zero(self):
        assert megalith.shards_needed(0) >= 1


class TestMegalithExtraction:
    def test_extracts_images_and_ignores_sidecars(self, settings, tmp_path):
        keys = [f"{i:09d}" for i in range(5)]
        tar_path = tmp_path / "00000.tar"
        build_shard_tar(tar_path, [(k, JPEG_BYTES + k.encode()) for k in keys])
        manifest = Manifest.load(tmp_path / "m.parquet", megalith.SOURCE)
        stats = FetchStats()

        extracted = megalith._extract_shard(
            settings,
            manifest,
            stats,
            tar_path=str(tar_path),
            shard=0,
            metadata=shard_metadata(keys),
            remaining=10,
            max_bytes=8 * 1024**2,
        )

        assert extracted == 5
        assert stats.downloaded == 5
        assert manifest.succeeded == 5
        stored = list((settings.originals_dir / megalith.SOURCE).rglob("*.jpg"))
        assert len(stored) == 5

    def test_stores_files_content_addressed_under_the_source(self, settings, tmp_path):
        data = JPEG_BYTES + b"unique"
        tar_path = tmp_path / "00000.tar"
        build_shard_tar(tar_path, [("000000001", data)])
        manifest = Manifest.load(tmp_path / "m.parquet", megalith.SOURCE)

        megalith._extract_shard(
            settings,
            manifest,
            FetchStats(),
            tar_path=str(tar_path),
            shard=0,
            metadata=shard_metadata(["000000001"]),
            remaining=10,
            max_bytes=8 * 1024**2,
        )

        expected = settings.original_path(megalith.SOURCE, image_id_for(data), ".jpg")
        assert expected.read_bytes() == data

    def test_records_the_flickr_page_as_the_source_url(self, settings, tmp_path):
        # The mirror's `caption` column holds the photo page URL, which is the
        # provenance link a human needs to check an individual licence.
        tar_path = tmp_path / "00000.tar"
        build_shard_tar(tar_path, [("000000001", JPEG_BYTES)])
        manifest = Manifest.load(tmp_path / "m.parquet", megalith.SOURCE)

        megalith._extract_shard(
            settings,
            manifest,
            FetchStats(),
            tar_path=str(tar_path),
            shard=0,
            metadata=shard_metadata(["000000001"]),
            remaining=10,
            max_bytes=8 * 1024**2,
        )

        candidate = manifest.candidates["00000/000000001"]
        assert candidate.source_url == "https://www.flickr.com/photos/user/000000001"
        assert candidate.license_url == megalith.LICENCE_STATEMENT_URL

    def test_stops_at_the_remaining_count(self, settings, tmp_path):
        keys = [f"{i:09d}" for i in range(10)]
        tar_path = tmp_path / "00000.tar"
        build_shard_tar(tar_path, [(k, JPEG_BYTES + k.encode()) for k in keys])
        manifest = Manifest.load(tmp_path / "m.parquet", megalith.SOURCE)

        extracted = megalith._extract_shard(
            settings,
            manifest,
            FetchStats(),
            tar_path=str(tar_path),
            shard=0,
            metadata=shard_metadata(keys),
            remaining=3,
            max_bytes=8 * 1024**2,
        )

        assert extracted == 3

    def test_re_running_extracts_nothing_new(self, settings, tmp_path):
        # The Phase 2 Definition of Done: "rerun downloads nothing new".
        keys = [f"{i:09d}" for i in range(4)]
        tar_path = tmp_path / "00000.tar"
        build_shard_tar(tar_path, [(k, JPEG_BYTES + k.encode()) for k in keys])
        manifest = Manifest.load(tmp_path / "m.parquet", megalith.SOURCE)
        metadata = shard_metadata(keys)

        first = megalith._extract_shard(
            settings,
            manifest,
            FetchStats(),
            tar_path=str(tar_path),
            shard=0,
            metadata=metadata,
            remaining=10,
            max_bytes=8 * 1024**2,
        )
        second_stats = FetchStats()
        second = megalith._extract_shard(
            settings,
            manifest,
            second_stats,
            tar_path=str(tar_path),
            shard=0,
            metadata=metadata,
            remaining=10,
            max_bytes=8 * 1024**2,
        )

        assert first == 4
        assert second == 0, "a completed shard must not be re-extracted"
        assert second_stats.requested == 0
        assert manifest.succeeded == 4

    def test_resumes_a_shard_interrupted_half_way(self, settings, tmp_path):
        keys = [f"{i:09d}" for i in range(6)]
        tar_path = tmp_path / "00000.tar"
        build_shard_tar(tar_path, [(k, JPEG_BYTES + k.encode()) for k in keys])
        manifest = Manifest.load(tmp_path / "m.parquet", megalith.SOURCE)
        metadata = shard_metadata(keys)

        # A run killed after 2 images.
        megalith._extract_shard(
            settings,
            manifest,
            FetchStats(),
            tar_path=str(tar_path),
            shard=0,
            metadata=metadata,
            remaining=2,
            max_bytes=8 * 1024**2,
        )
        manifest.flush(force=True)

        # A fresh process reloads the manifest and picks up the other 4.
        resumed = Manifest.load(tmp_path / "m.parquet", megalith.SOURCE)
        extracted = megalith._extract_shard(
            settings,
            resumed,
            FetchStats(),
            tar_path=str(tar_path),
            shard=0,
            metadata=metadata,
            remaining=10,
            max_bytes=8 * 1024**2,
        )

        assert extracted == 4
        assert resumed.succeeded == 6
        stored = list((settings.originals_dir / megalith.SOURCE).rglob("*.jpg"))
        assert len(stored) == 6, "resume must not duplicate files"

    def test_rejects_an_oversized_member(self, settings, tmp_path):
        tar_path = tmp_path / "00000.tar"
        build_shard_tar(tar_path, [("000000001", JPEG_BYTES * 500)])
        manifest = Manifest.load(tmp_path / "m.parquet", megalith.SOURCE)
        stats = FetchStats()

        extracted = megalith._extract_shard(
            settings,
            manifest,
            stats,
            tar_path=str(tar_path),
            shard=0,
            metadata=shard_metadata(["000000001"]),
            remaining=10,
            max_bytes=1000,
        )

        assert extracted == 0
        assert stats.rejected == {"too_large": 1}

    def test_rejects_a_member_whose_declared_shape_is_wrong(self, settings, tmp_path):
        tar_path = tmp_path / "00000.tar"
        build_shard_tar(tar_path, [("000000001", JPEG_BYTES)])
        manifest = Manifest.load(tmp_path / "m.parquet", megalith.SOURCE)
        metadata = shard_metadata(["000000001"])
        metadata["000000001"]["height"] = 100.0  # 1024x100 is past 4:1
        stats = FetchStats()

        extracted = megalith._extract_shard(
            settings,
            manifest,
            stats,
            tar_path=str(tar_path),
            shard=0,
            metadata=metadata,
            remaining=10,
            max_bytes=8 * 1024**2,
        )

        assert extracted == 0
        assert stats.rejected == {"bad_dimensions": 1}

    def test_skips_members_absent_from_the_shard_metadata(self, settings, tmp_path):
        # The mirror's parquet marks failed downloads; a tar member with no
        # successful row behind it has no provenance, so it is not taken.
        tar_path = tmp_path / "00000.tar"
        build_shard_tar(tar_path, [("000000001", JPEG_BYTES), ("000000002", JPEG_BYTES)])
        manifest = Manifest.load(tmp_path / "m.parquet", megalith.SOURCE)

        extracted = megalith._extract_shard(
            settings,
            manifest,
            FetchStats(),
            tar_path=str(tar_path),
            shard=0,
            metadata=shard_metadata(["000000001"]),
            remaining=10,
            max_bytes=8 * 1024**2,
        )

        assert extracted == 1


# --------------------------------------------------------------------------
# PD12M
# --------------------------------------------------------------------------


def pd12m_frame(rows):
    return pd.DataFrame(rows)


class TestPd12mNormalisation:
    def test_missing_values_become_none_not_the_string_nan(self):
        # pandas 3 keeps NaN through `.astype(str)`, and `bool(nan)` is True, so
        # the obvious `str(x) if x else None` yields "nan". As an expected MD5
        # that would make every image fail verification.
        assert pd12m._text(float("nan")) is None
        assert pd12m._text(None) is None
        assert pd12m._text("  ") is None
        assert pd12m._text(" Albertina ") == "Albertina"

    def test_numbers_handle_nan_without_raising(self):
        assert pd12m._number(float("nan")) is None
        assert pd12m._number(None) is None
        assert pd12m._number(1024.0) == 1024

    def test_captions_are_truncated_rather_than_stored_whole(self):
        long_caption = "The image shows " + "a very long description " * 50

        result = pd12m._short_caption(long_caption)

        assert result is not None
        assert len(result) == 200
        assert result.endswith("...")

    def test_a_blank_caption_becomes_none(self):
        assert pd12m._short_caption("") is None
        assert pd12m._short_caption(None) is None
        assert pd12m._short_caption(float("nan")) is None


class TestPd12mGates:
    def test_rows_without_a_licence_url_are_dropped(self):
        frame = pd12m_frame(
            [
                {
                    "id": "a",
                    "url": "u1",
                    "license": "https://cc0",
                    "mime_type": "image/jpeg",
                    "width": 800,
                    "height": 600,
                    "hash": "h",
                    "source": "Museum",
                    "caption": None,
                },
                {
                    "id": "b",
                    "url": "u2",
                    "license": None,
                    "mime_type": "image/jpeg",
                    "width": 800,
                    "height": 600,
                    "hash": "h",
                    "source": "Museum",
                    "caption": None,
                },
            ]
        )

        result = pd12m._apply_gates(frame)

        assert list(result["id"]) == ["a"]

    def test_unwanted_containers_are_dropped(self):
        frame = pd12m_frame(
            [
                {
                    "id": "a",
                    "url": "u1",
                    "license": "https://cc0",
                    "mime_type": "image/tiff",
                    "width": 800,
                    "height": 600,
                    "hash": "h",
                    "source": "M",
                    "caption": None,
                },
                {
                    "id": "b",
                    "url": "u2",
                    "license": "https://cc0",
                    "mime_type": "image/jpeg",
                    "width": 800,
                    "height": 600,
                    "hash": "h",
                    "source": "M",
                    "caption": None,
                },
            ]
        )

        assert list(pd12m._apply_gates(frame)["id"]) == ["b"]

    def test_the_same_url_under_two_institutions_is_kept_once(self):
        frame = pd12m_frame(
            [
                {
                    "id": "a",
                    "url": "same",
                    "license": "https://cc0",
                    "mime_type": "image/jpeg",
                    "width": 800,
                    "height": 600,
                    "hash": "h",
                    "source": "M1",
                    "caption": None,
                },
                {
                    "id": "b",
                    "url": "same",
                    "license": "https://cc0",
                    "mime_type": "image/jpeg",
                    "width": 800,
                    "height": 600,
                    "hash": "h",
                    "source": "M2",
                    "caption": None,
                },
            ]
        )

        assert len(pd12m._apply_gates(frame)) == 1


class TestPd12mStratification:
    def _lopsided_frame(self):
        """1,000 rows where one institution holds 90% of them."""
        rows = []
        for i in range(900):
            rows.append({"id": f"big{i}", "source": "Wikimedia Commons"})
        for i in range(50):
            rows.append({"id": f"a{i}", "source": "Small Museum A"})
        for i in range(50):
            rows.append({"id": f"b{i}", "source": "Small Museum B"})
        return pd.DataFrame(rows)

    def test_a_dominant_institution_does_not_dominate_the_sample(self):
        frame = self._lopsided_frame()

        chosen = pd12m._round_robin_by_institution(frame, target=100, seed=1729)
        shares = chosen["source"].value_counts(normalize=True)

        # Unstratified, Wikimedia would be ~90%. Round-robin across three
        # institutions brings it to roughly a third.
        assert shares["Wikimedia Commons"] < 0.45

    def test_small_institutions_are_fully_represented(self):
        frame = self._lopsided_frame()

        chosen = pd12m._round_robin_by_institution(frame, target=100, seed=1729)

        assert set(chosen["source"]) == {
            "Wikimedia Commons",
            "Small Museum A",
            "Small Museum B",
        }

    def test_the_same_seed_gives_the_same_sample(self):
        frame = self._lopsided_frame()

        first = pd12m._round_robin_by_institution(frame, target=60, seed=1729)
        second = pd12m._round_robin_by_institution(frame, target=60, seed=1729)

        assert list(first["id"]) == list(second["id"])

    def test_a_different_seed_gives_a_different_sample(self):
        frame = self._lopsided_frame()

        first = pd12m._round_robin_by_institution(frame, target=60, seed=1729)
        other = pd12m._round_robin_by_institution(frame, target=60, seed=5)

        assert list(first["id"]) != list(other["id"])

    def test_rows_with_no_institution_get_their_own_bucket(self):
        frame = pd.DataFrame(
            [{"id": f"x{i}", "source": None} for i in range(10)]
            + [{"id": f"y{i}", "source": "A Museum"} for i in range(10)]
        )

        chosen = pd12m._round_robin_by_institution(frame, target=20, seed=1729)

        assert len(chosen) == 20

    def test_asking_for_more_than_exists_returns_everything(self):
        frame = self._lopsided_frame()

        chosen = pd12m._round_robin_by_institution(frame, target=100_000, seed=1729)

        assert len(chosen) == len(frame)


# --------------------------------------------------------------------------
# Open Images
# --------------------------------------------------------------------------


def oi_row(**overrides):
    row = {
        "ImageID": "fe600639ac5f36c1",
        "Subset": "validation",
        "OriginalURL": "https://farm2.staticflickr.example/x_o.jpg",
        "OriginalLandingURL": "https://www.flickr.com/photos/118815643@N04/15340259497",
        "License": "https://creativecommons.org/licenses/by/2.0/",
        "AuthorProfileURL": "https://www.flickr.com/people/118815643@N04/",
        "Author": "LabHacker CD",
        "Title": "_GUT6674",
        "OriginalSize": "242145",
        "OriginalMD5": "0jBpbNION09+r02xkTIBcA==",
        "Rotation": "0.0",
    }
    row.update(overrides)
    return row


class TestOpenImagesGates:
    def test_an_ordinary_row_is_usable(self):
        assert openimages._is_usable(oi_row()) is True

    def test_a_row_without_a_licence_is_dropped(self):
        assert openimages._is_usable(oi_row(License="")) is False

    def test_a_row_without_an_author_is_dropped(self):
        # CC BY requires attribution; we will not display it uncredited.
        assert openimages._is_usable(oi_row(Author="")) is False
        assert openimages._is_usable(oi_row(Author="   ")) is False

    def test_blank_and_zero_rotation_are_both_accepted(self):
        assert openimages._is_usable(oi_row(Rotation="")) is True
        assert openimages._is_usable(oi_row(Rotation="0")) is True
        assert openimages._is_usable(oi_row(Rotation="0.0")) is True

    def test_a_rotated_row_is_skipped(self):
        # Correcting Open Images' own rotation field is not part of this
        # project's ingestion rules, and the split has far more candidates than
        # we need, so these are skipped rather than silently stored sideways.
        assert openimages._is_usable(oi_row(Rotation="90.0")) is False


class TestOpenImagesAttribution:
    def test_the_candidate_carries_everything_cc_by_requires(self):
        candidate = openimages._to_candidate(oi_row())

        assert candidate.license_url == "https://creativecommons.org/licenses/by/2.0/"
        assert candidate.author == "LabHacker CD"
        assert candidate.author_url == "https://www.flickr.com/people/118815643@N04/"
        assert candidate.source_url.startswith("https://www.flickr.com/photos/")

    def test_the_url_points_at_the_official_s3_redistribution(self):
        candidate = openimages._to_candidate(oi_row())

        assert candidate.url == "s3://open-images-dataset/validation/fe600639ac5f36c1.jpg"

    def test_the_upstream_digest_is_kept_but_not_used_as_a_verifier(self):
        # The S3 object is a re-encoded copy, so verifying against the Flickr
        # original's digest would reject every image (DECISIONS.md #7).
        candidate = openimages._to_candidate(oi_row())

        assert candidate.expected_md5 is None
        assert candidate.upstream_md5 == "d230696cd20e374f7eaf4db191320170"

    def test_base64_digests_are_decoded_to_hex(self):
        assert openimages.original_md5_hex(oi_row()) == "d230696cd20e374f7eaf4db191320170"

    def test_a_missing_or_malformed_digest_is_none_rather_than_an_error(self):
        assert openimages.original_md5_hex(oi_row(OriginalMD5="")) is None
        assert openimages.original_md5_hex(oi_row(OriginalMD5="not base64 !!")) is None


# --------------------------------------------------------------------------
# Unsplash Lite
# --------------------------------------------------------------------------


def unsplash_row(**overrides):
    row = {
        "photo_id": "vnY8Xl4RhD4",
        "photo_url": "https://unsplash.com/photos/vnY8Xl4RhD4",
        "photo_image_url": "https://images.unsplash.com/photo-1234567890",
        "photographer_name": "Daiga Ellaby",
        "photographer_username": "daiga_ellaby",
        "photo_description": "A calm interior",
        "photo_width": "4000",
        "photo_height": "3000",
    }
    row.update(overrides)
    return row


class TestUnsplashGates:
    def test_an_ordinary_row_is_usable(self):
        assert unsplash._is_usable(unsplash_row()) is True

    def test_a_row_without_an_image_url_is_dropped(self):
        assert unsplash._is_usable(unsplash_row(photo_image_url="")) is False

    def test_a_row_without_a_photographer_is_dropped(self):
        # The Unsplash terms ask us to credit the photographer, so a row we
        # cannot attribute is never downloaded.
        assert unsplash._is_usable(unsplash_row(photographer_username="")) is False


class TestUnsplashCandidates:
    def test_sizing_parameters_are_appended_to_the_image_url(self):
        candidate = unsplash._to_candidate(unsplash_row())

        assert candidate.url.endswith("?w=2048&q=85&fm=jpg")

    def test_an_existing_query_string_is_extended_not_replaced(self):
        candidate = unsplash._to_candidate(
            unsplash_row(photo_image_url="https://images.unsplash.com/photo-1?ixid=abc")
        )

        assert "ixid=abc" in candidate.url
        assert candidate.url.endswith("&w=2048&q=85&fm=jpg")

    def test_attribution_is_carried_through(self):
        candidate = unsplash._to_candidate(unsplash_row())

        assert candidate.author == "Daiga Ellaby"
        assert candidate.author_url == "https://unsplash.com/@daiga_ellaby"
        assert candidate.source_url == "https://unsplash.com/photos/vnY8Xl4RhD4"
        assert candidate.license_url == "https://unsplash.com/license"

    def test_a_missing_photographer_name_falls_back_to_the_username(self):
        candidate = unsplash._to_candidate(unsplash_row(photographer_name=""))

        assert candidate.author == "daiga_ellaby"

    def test_a_missing_photo_page_is_reconstructed_from_the_id(self):
        candidate = unsplash._to_candidate(unsplash_row(photo_url=""))

        assert candidate.source_url == "https://unsplash.com/photos/vnY8Xl4RhD4"

    def test_dimensions_are_parsed_from_the_tsv_strings(self):
        candidate = unsplash._to_candidate(unsplash_row())

        assert candidate.declared_width == 4000
        assert candidate.declared_height == 3000

    def test_unparseable_dimensions_become_none(self):
        candidate = unsplash._to_candidate(unsplash_row(photo_width="", photo_height="n/a"))

        assert candidate.declared_width is None
        assert candidate.declared_height is None


class TestUnsplashTsvReading:
    def test_reads_rows_out_of_the_dataset_zip(self, tmp_path):
        import zipfile

        zip_path = tmp_path / "unsplash.zip"
        tsv = (
            "photo_id\tphoto_url\tphoto_image_url\tphotographer_username\n"
            "abc\thttps://unsplash.com/photos/abc\thttps://images.unsplash.com/x\tsomeone\n"
        )
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("photos.tsv000", tsv)
            archive.writestr("keywords.tsv000", "irrelevant\n")

        rows = unsplash.read_photos_tsv(zip_path)

        assert len(rows) == 1
        assert rows[0]["photo_id"] == "abc"

    def test_a_description_containing_a_quote_does_not_eat_later_columns(self, tmp_path):
        # Unsplash descriptions contain unescaped quote characters, which the
        # default csv dialect treats as quoting and then swallows the rest of
        # the line. QUOTE_NONE is why this works.
        import zipfile

        zip_path = tmp_path / "unsplash.zip"
        tsv = 'photo_id\tphoto_description\tphotographer_username\nabc\tshe said "hello\tsomeone\n'
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("photos.tsv000", tsv)

        rows = unsplash.read_photos_tsv(zip_path)

        assert rows[0]["photographer_username"] == "someone"

    def test_a_zip_without_a_photos_table_fails_loudly(self, tmp_path):
        import zipfile

        zip_path = tmp_path / "unsplash.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("something_else.tsv", "x\n")

        with pytest.raises(RuntimeError, match=r"no photos\.tsv"):
            unsplash.read_photos_tsv(zip_path)
