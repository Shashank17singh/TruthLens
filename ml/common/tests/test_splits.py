import pandas as pd
import pytest

from ml.common.manifest import (
    FAKE,
    REAL,
    REAL_GENERATOR,
    LeakageError,
    ManifestError,
    Record,
    check_no_leakage,
    read_manifest,
    records_to_frame,
    unseen_evaluation_frame,
    validate,
    write_manifest,
)
from ml.common.splits import SplitConfig, assign_splits, connect_groups


def face_dataset(identities: int = 400) -> pd.DataFrame:
    """Every identity has real samples and one fake per generator, like a face-swap set."""
    records = []
    for identity in range(identities):
        for clip in range(3):
            records.append(
                Record(f"real/{identity}/{clip}", REAL, "toy", REAL_GENERATOR, f"id{identity}", content_hash=f"r{identity}-{clip}")
            )
        for generator in ("gan_a", "gan_b", "diffusion_a"):
            records.append(
                Record(f"{generator}/{identity}", FAKE, "toy", generator, f"id{identity}", content_hash=f"{generator}-{identity}")
            )
    return records_to_frame(records)


def test_no_file_or_content_appears_in_two_splits():
    result = assign_splits(face_dataset(), SplitConfig())
    frame = result.frame
    assert frame.groupby("path")["split"].nunique().max() == 1
    assert frame.groupby("content_hash")["split"].nunique().max() == 1


def test_no_identity_appears_in_two_splits():
    frame = assign_splits(face_dataset(), SplitConfig()).frame
    assert frame.groupby("group_id")["split"].nunique().max() == 1
    check_no_leakage(frame)


def test_ratios_are_close_to_target():
    frame = assign_splits(face_dataset(), SplitConfig()).frame
    shares = frame["split"].value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.02)
    assert shares["val"] == pytest.approx(0.15, abs=0.02)
    assert shares["test"] == pytest.approx(0.15, abs=0.02)


def test_each_split_keeps_label_and_generator_mix():
    frame = assign_splits(face_dataset(), SplitConfig()).frame
    for _, rows in frame.groupby("split"):
        shares = rows["generator_name"].value_counts(normalize=True)
        assert shares[REAL_GENERATOR] == pytest.approx(0.5, abs=0.02)
        assert shares["gan_a"] == pytest.approx(1 / 6, abs=0.02)


def test_rare_generator_reaches_every_split():
    records = [
        Record(f"r{i}", REAL, "toy", REAL_GENERATOR, f"g{i}", content_hash=f"r{i}") for i in range(2000)
    ] + [Record(f"f{i}", FAKE, "toy", "rare", f"h{i}", content_hash=f"f{i}") for i in range(40)]
    frame = assign_splits(records_to_frame(records), SplitConfig()).frame
    counts = frame[frame["generator_name"] == "rare"]["split"].value_counts()
    assert counts["train"] == pytest.approx(28, abs=2)
    assert counts["val"] == pytest.approx(6, abs=2)
    assert counts["test"] == pytest.approx(6, abs=2)


def test_split_is_repeatable_and_depends_on_seed():
    first = assign_splits(face_dataset(), SplitConfig(seed=1)).frame
    again = assign_splits(face_dataset(), SplitConfig(seed=1)).frame
    other = assign_splits(face_dataset(), SplitConfig(seed=2)).frame
    pd.testing.assert_frame_equal(first, again)
    assert not first["split"].equals(other["split"])


def test_unseen_generator_never_reaches_training():
    config = SplitConfig(heldout_generators=frozenset({"diffusion_a"}))
    frame = assign_splits(face_dataset(), config).frame
    assert set(frame.loc[frame["generator_name"] == "diffusion_a", "split"]) == {"unseen_test"}
    assert (frame["split"] == "unseen_test").all() == False  # noqa: E712
    check_no_leakage(frame)


def test_unseen_samples_never_share_an_identity_with_training():
    config = SplitConfig(heldout_generators=frozenset({"diffusion_a"}))
    result = assign_splits(face_dataset(), config)
    frame = result.frame
    unseen = set(frame.loc[frame["split"] == "unseen_test", "group_id"])
    fitted = set(frame.loc[frame["split"].isin(["train", "val"]), "group_id"])
    assert unseen and not unseen & fitted
    assert result.dropped_heldout > 0


def test_group_of_only_heldout_rows_is_fully_kept():
    records = [Record(f"r{i}", REAL, "toy", REAL_GENERATOR, f"r{i}", content_hash=f"r{i}") for i in range(100)]
    records += [Record(f"h{i}", FAKE, "toy", "held", f"h{i}", content_hash=f"h{i}") for i in range(50)]
    result = assign_splits(records_to_frame(records), SplitConfig(heldout_generators=frozenset({"held"})))
    assert (result.frame["split"] == "unseen_test").sum() == 50
    assert result.dropped_heldout == 0


def test_official_split_is_preserved_and_heldout_moves_to_unseen():
    records = [
        Record("a", REAL, "toy", REAL_GENERATOR, "s1", content_hash="a", split="train"),
        Record("b", FAKE, "toy", "A01", "s1", content_hash="b", split="train"),
        Record("c", REAL, "toy", REAL_GENERATOR, "s2", content_hash="c", split="val"),
        Record("d", FAKE, "toy", "A01", "s3", content_hash="d", split="test"),
        Record("e", FAKE, "toy", "A07", "s3", content_hash="e", split="test"),
    ]
    config = SplitConfig(strategy="official", heldout_generators=frozenset({"A07"}))
    frame = assign_splits(records_to_frame(records), config).frame
    assert dict(zip(frame["path"], frame["split"])) == {
        "a": "train", "b": "train", "c": "val", "d": "test", "e": "unseen_test",
    }


def test_official_strategy_drops_heldout_rows_from_training():
    records = [
        Record("a", FAKE, "toy", "held", "s1", content_hash="a", split="train"),
        Record("b", FAKE, "toy", "held", "s2", content_hash="b", split="test"),
    ]
    result = assign_splits(
        records_to_frame(records), SplitConfig(strategy="official", heldout_generators=frozenset({"held"}))
    )
    assert list(result.frame["path"]) == ["b"]
    assert result.dropped_heldout == 1


def test_official_strategy_refuses_missing_split():
    records = [Record("a", REAL, "toy", REAL_GENERATOR, "s1", content_hash="a")]
    with pytest.raises(ValueError, match="official"):
        assign_splits(records_to_frame(records), SplitConfig(strategy="official"))


def test_evaluation_strategy_puts_everything_in_test():
    frame = assign_splits(face_dataset(5), SplitConfig(strategy="evaluation")).frame
    assert set(frame["split"]) == {"test"}


def test_too_few_groups_is_an_error():
    records = [Record("a", REAL, "toy", REAL_GENERATOR, "only", content_hash="a")]
    with pytest.raises(ValueError, match="groups"):
        assign_splits(records_to_frame(records), SplitConfig())


def test_connect_groups_links_chains():
    groups = connect_groups([("1", "2"), ("2", "3"), ("4", "5")])
    assert groups["1"] == groups["2"] == groups["3"]
    assert groups["4"] == groups["5"]
    assert groups["1"] != groups["4"]


def test_leakage_check_catches_a_shared_path_hash_or_group():
    base = {"label": REAL, "source_dataset": "toy", "generator_name": REAL_GENERATOR, "domain": ""}
    shared_hash = pd.DataFrame([
        {**base, "path": "a", "group_id": "g1", "content_hash": "same", "split": "train"},
        {**base, "path": "b", "group_id": "g2", "content_hash": "same", "split": "test"},
    ])
    with pytest.raises(LeakageError, match="identical content"):
        check_no_leakage(shared_hash)

    shared_group = pd.DataFrame([
        {**base, "path": "a", "group_id": "g", "content_hash": "h1", "split": "train"},
        {**base, "path": "b", "group_id": "g", "content_hash": "h2", "split": "val"},
    ])
    with pytest.raises(LeakageError, match="a group"):
        check_no_leakage(shared_group)

    duplicate_path = pd.DataFrame([
        {**base, "path": "a", "group_id": "g1", "content_hash": "h1", "split": "train"},
        {**base, "path": "a", "group_id": "g2", "content_hash": "h2", "split": "test"},
    ])
    with pytest.raises(ManifestError, match="once"):
        validate(duplicate_path)


def test_leakage_check_catches_unseen_generator_in_training():
    frame = records_to_frame([
        Record("a", FAKE, "toy", "gen", "g1", content_hash="a", split="train"),
        Record("b", FAKE, "toy", "gen", "g2", content_hash="b", split="unseen_test"),
    ])
    with pytest.raises(LeakageError, match="unseen generators"):
        check_no_leakage(frame)


def test_leakage_check_catches_contradictory_labels():
    frame = records_to_frame([
        Record("a", REAL, "toy", REAL_GENERATOR, "g", content_hash="same", split="test"),
        Record("b", FAKE, "toy", "gen", "g", content_hash="same", split="test"),
    ])
    with pytest.raises(LeakageError, match="both labels"):
        check_no_leakage(frame)


def test_validate_rejects_real_label_with_generator():
    frame = records_to_frame([Record("a", REAL, "toy", "gen", "g", content_hash="h", split="test")])
    with pytest.raises(ManifestError, match="generator_name"):
        validate(frame)


def test_manifest_round_trip_keeps_values_and_sorts(tmp_path):
    frame = assign_splits(face_dataset(10), SplitConfig()).frame
    path = tmp_path / "toy.csv"
    write_manifest(frame.sample(frac=1, random_state=0), path)
    loaded = read_manifest(path)
    assert list(loaded["path"]) == sorted(frame["path"])
    assert len(loaded) == len(frame)


def test_write_manifest_refuses_leaky_frame(tmp_path):
    frame = records_to_frame([
        Record("a", REAL, "toy", REAL_GENERATOR, "g", content_hash="h1", split="train"),
        Record("b", REAL, "toy", REAL_GENERATOR, "g", content_hash="h2", split="test"),
    ])
    with pytest.raises(LeakageError):
        write_manifest(frame, tmp_path / "bad.csv")
    assert not (tmp_path / "bad.csv").exists()


def test_unseen_evaluation_frame_has_both_classes():
    config = SplitConfig(heldout_generators=frozenset({"diffusion_a"}))
    frame = assign_splits(face_dataset(), config).frame
    evaluation = unseen_evaluation_frame(frame)
    assert set(evaluation["label"]) == {REAL, FAKE}
    assert set(evaluation.loc[evaluation["label"] == FAKE, "generator_name"]) == {"diffusion_a"}
