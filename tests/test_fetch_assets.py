from scripts.fetch_assets import _first_file


def test_first_file_skips_download_metadata_and_cache(tmp_path):
    (tmp_path / "download_meta.json").write_text("{}")
    cache = tmp_path / ".cache"
    cache.mkdir()
    (cache / "metadata.json").write_text("{}")
    dataset = tmp_path / "test.jsonl"
    dataset.write_text('{"problem": "1+1", "answer": "2"}\n')

    assert _first_file(tmp_path, (".parquet", ".jsonl", ".json")) == dataset


def test_first_file_returns_none_for_metadata_only(tmp_path):
    (tmp_path / "download_meta.json").write_text("{}")

    assert _first_file(tmp_path, (".parquet", ".jsonl", ".json")) is None


def test_first_file_prefers_parquet_over_root_json(tmp_path):
    (tmp_path / "aaa.json").write_text("{}")
    parquet = tmp_path / "data.parquet"
    parquet.write_bytes(b"PAR1")
    assert _first_file(tmp_path, (".parquet", ".jsonl", ".json")) == parquet
