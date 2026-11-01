import http.server
import json
import tarfile
import threading
import zipfile
from datetime import date
from functools import partial

import numpy as np
import pandas as pd
import pytest

from ml.common import balance, datacard
from ml.common.checksums import ChecksumError, hash_file, verify_file, verify_or_record
from ml.common.download import download, extract
from ml.common.manifest import FAKE, REAL, REAL_GENERATOR, Record, combine, records_to_frame, write_manifest
from ml.common.prepare import REGISTRY_PATH, load_yaml, stratified_limit


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        body = (self.directory_path / self.path.lstrip("/")).read_bytes()
        requested = self.headers.get("Range")
        if requested:
            offset = int(requested.removeprefix("bytes=").removesuffix("-"))
            self.send_response(206)
            body = body[offset:]
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def file_server(tmp_path):
    served = tmp_path / "served"
    served.mkdir()
    handler = partial(RangeHandler, directory=str(served))
    RangeHandler.directory_path = served
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield served, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_verify_file_accepts_match_and_rejects_change(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"truthlens")
    verify_file(path, f"sha256:{hash_file(path)}")
    verify_file(path, f"md5:{hash_file(path, 'md5')}")
    path.write_bytes(b"tampered")
    with pytest.raises(ChecksumError):
        verify_file(path, "sha256:" + "0" * 64)


def test_lock_records_once_then_detects_a_changed_file(tmp_path):
    path = tmp_path / "a.bin"
    lock = tmp_path / "lock.json"
    path.write_bytes(b"first")
    assert verify_or_record(path, "toy/a.bin", lock) is True
    assert verify_or_record(path, "toy/a.bin", lock) is False
    path.write_bytes(b"second")
    with pytest.raises(ChecksumError):
        verify_or_record(path, "toy/a.bin", lock)


def test_download_verifies_and_resumes(file_server, tmp_path):
    served, url = file_server
    payload = bytes(range(256)) * 4096
    (served / "data.bin").write_bytes(payload)
    target = tmp_path / "out" / "data.bin"
    target.parent.mkdir()
    (tmp_path / "out" / "data.bin.part").write_bytes(payload[:1000])

    expected = f"sha256:{hash_file(served / 'data.bin')}"
    download(f"{url}/data.bin", target, expected)
    assert target.read_bytes() == payload


def test_download_removes_corrupt_file(file_server, tmp_path):
    served, url = file_server
    (served / "data.bin").write_bytes(b"wrong bytes")
    target = tmp_path / "data.bin"
    with pytest.raises(ChecksumError):
        download(f"{url}/data.bin", target, "sha256:" + "0" * 64)
    assert not target.exists()
    assert not target.with_name("data.bin.part").exists()


def test_extract_runs_once(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("folder/x.txt", "x")
    out = tmp_path / "out"
    extract(archive, out)
    assert (out / "folder" / "x.txt").read_text() == "x"
    (out / "folder" / "x.txt").unlink()
    extract(archive, out)
    assert not (out / "folder" / "x.txt").exists()


def test_extract_refuses_tar_path_traversal(tmp_path):
    payload = tmp_path / "payload.txt"
    payload.write_text("x")
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(payload, arcname="../escaped.txt")
    with pytest.raises(tarfile.FilterError):
        extract(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def imbalanced_frame() -> pd.DataFrame:
    records = [Record(f"r{i}", REAL, "toy", REAL_GENERATOR, f"r{i}", content_hash=f"r{i}", split="train") for i in range(100)]
    records += [Record(f"a{i}", FAKE, "toy", "big", f"a{i}", content_hash=f"a{i}", split="train") for i in range(880)]
    records += [Record(f"b{i}", FAKE, "toy", "small", f"b{i}", content_hash=f"b{i}", split="train") for i in range(20)]
    return records_to_frame(records)


def test_balance_report_counts_splits_and_generators():
    report = balance.balance_report(imbalanced_frame())
    assert report["total"] == 1000
    assert report["splits"]["train"] == {
        "samples": 1000, "real": 100, "fake": 900, "fake_share": 0.9, "groups": 1000,
    }
    assert report["generators"]["small"]["train"] == 20


def test_sample_weights_balance_labels_and_generators():
    frame = imbalanced_frame()
    weights = balance.sample_weights(frame)
    assert weights.sum() == pytest.approx(1.0)
    share = pd.Series(weights).groupby(frame["generator_name"].to_numpy()).sum()
    assert share[REAL_GENERATOR] == pytest.approx(0.5)
    assert share["big"] == pytest.approx(0.25)
    assert share["small"] == pytest.approx(0.25)


def test_weighted_sampler_is_seeded_and_balances_draws():
    frame = imbalanced_frame()
    first = list(balance.weighted_sampler(frame, seed=7))
    second = list(balance.weighted_sampler(frame, seed=7))
    assert first == second
    drawn = frame.iloc[first]["label"].mean()
    assert drawn == pytest.approx(0.5, abs=0.05)


def test_combine_detects_leakage_between_datasets(tmp_path):
    one = records_to_frame([Record("a/1", REAL, "a", REAL_GENERATOR, "x", content_hash="same", split="train")])
    two = records_to_frame([Record("b/1", REAL, "b", REAL_GENERATOR, "y", content_hash="same", split="test")])
    write_manifest(one, tmp_path / "one.csv")
    write_manifest(two, tmp_path / "two.csv")
    with pytest.raises(ValueError, match="identical content"):
        combine([tmp_path / "one.csv", tmp_path / "two.csv"])


def test_stratified_limit_keeps_whole_groups():
    records = []
    for group in range(100):
        records.append(Record(f"r{group}", REAL, "toy", REAL_GENERATOR, f"g{group}"))
        records.append(Record(f"f{group}", FAKE, "toy", "gen", f"g{group}"))
    subset = stratified_limit(records, 40, seed=1)
    assert 0 < len(subset) < len(records)
    frame = records_to_frame(subset)
    assert (frame.groupby("group_id").size() == 2).all()


def test_data_card_has_the_required_sections():
    entry = {
        "title": "Toy", "modality": "text", "role": "Testing", "licence": "MIT",
        "access": "open", "homepage": "https://example.com", "known_biases": ["Small sample."],
    }
    report = balance.balance_report(imbalanced_frame())
    card = datacard.render("toy", entry, report, {"input_records": 1000}, date(2026, 11, 1))
    for expected in ("## Summary", "## Classes and splits", "## Sources and generators", "## Known biases", "MIT", "Small sample.", "1,000"):
        assert expected in card


def test_every_registry_entry_is_complete():
    registry = load_yaml(REGISTRY_PATH)
    required = {"title", "modality", "adapter", "role", "licence", "access", "homepage", "split", "known_biases"}
    for name, entry in registry.items():
        assert required <= set(entry), name
        assert entry["modality"] in {"image", "audio", "text"}, name
        assert entry["access"] in {"open", "manual"}, name
        assert entry["split"]["strategy"] in {"grouped", "official", "evaluation"}, name
        assert entry["known_biases"], name
        if entry["access"] == "manual":
            assert entry["instructions"], name
        else:
            assert entry["files"], name


def test_asvspoof_heldout_excludes_reused_attacks():
    held = set(load_yaml(REGISTRY_PATH)["asvspoof2019_la"]["split"]["heldout_generators"])
    assert not held & {"A16", "A19"}
    assert len(held) == 11


def test_seeded_numpy_is_repeatable():
    from ml.common.seed import set_seed

    set_seed(5)
    first = np.random.rand(3)
    set_seed(5)
    assert (first == np.random.rand(3)).all()


def test_manifest_balance_json_is_serializable():
    assert json.dumps(balance.balance_report(imbalanced_frame()))
