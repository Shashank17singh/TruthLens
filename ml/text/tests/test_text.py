import json

import pandas as pd
import pytest

from ml.common.manifest import FAKE, REAL, REAL_GENERATOR
from ml.text import datasets, preprocess
from ml.common.manifest import Record

CONFIG = {"min_words": 50, "max_words": 1000}
ENGLISH = (
    "The committee reviewed the proposal carefully and decided that more evidence was needed "
    "before any final decision could be made about the future of the program. Several members "
    "argued that the budget should be increased, while others believed the money would be better "
    "spent on training teachers and improving the library for every student in the district today."
)
HINDI = " ".join(["यह एक बहुत अच्छी किताब है और मुझे इसे पढ़ना पसंद है"] * 8)


def record(path: str, text: str, label: int = REAL, split: str = "") -> Record:
    generator = REAL_GENERATOR if label == REAL else "gen"
    return Record(path, label, "toy", generator, path, split=split, text=text)


def test_normalize_removes_tokenizer_spacing():
    raw = 'Basically there are many categories of " Best Seller " . Replace it , do n\'t you ( yes ) ?'
    assert preprocess.normalize(raw) == 'Basically there are many categories of "Best Seller". Replace it, don\'t you (yes)?'


def test_normalize_removes_zero_width_characters_and_extra_space():
    assert preprocess.normalize("a​ b  \n c") == "a b c"


def test_content_hash_ignores_case_and_punctuation():
    assert preprocess.content_hash("Hello, World!") == preprocess.content_hash("hello world")
    assert preprocess.content_hash("hello world") != preprocess.content_hash("hello there")


def test_prepare_applies_length_filter_at_both_ends(tmp_path):
    words = "word " * 1001
    kept, stats = preprocess.prepare(
        [record("short", "too short"), record("long", words), record("ok", ENGLISH)], tmp_path, CONFIG
    )
    assert len(kept) == 1
    assert stats["dropped_length"] == 2


def test_prepare_filters_non_english(tmp_path):
    kept, stats = preprocess.prepare([record("en", ENGLISH), record("hi", HINDI)], tmp_path, CONFIG)
    assert len(kept) == 1
    assert stats["dropped_language"] == 1


def test_prepare_removes_duplicates_and_keeps_the_test_copy(tmp_path):
    kept, stats = preprocess.prepare(
        [record("a", ENGLISH, split="train"), record("b", ENGLISH.upper().lower(), split="test")],
        tmp_path,
        CONFIG,
    )
    assert [item.split for item in kept] == ["test"]
    assert stats["dropped_duplicates"] == 1


def test_prepare_drops_text_with_contradictory_labels(tmp_path):
    kept, stats = preprocess.prepare([record("a", ENGLISH), record("b", ENGLISH, FAKE)], tmp_path, CONFIG)
    assert kept == []
    assert stats["dropped_conflicting_labels"] == 2


def test_prepare_writes_text_outside_the_records(tmp_path):
    kept, _ = preprocess.prepare([record("a", ENGLISH)], tmp_path, CONFIG)
    stored = pd.read_parquet(tmp_path / "texts.parquet")
    assert kept[0].text is None
    assert stored.loc[0, "path"] == kept[0].path
    assert stored.loc[0, "text"].startswith("The committee")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("cmv_human", ("cmv", REAL_GENERATOR)),
        ("sci_gen_human", ("sci_gen", REAL_GENERATOR)),
        ("eli5_machine_specified_gpt-3.5-trubo", ("eli5", "gpt-3.5-trubo")),
        ("xsum_machine_continuation_opt_2.7b", ("xsum", "opt_2.7b")),
        ("sci_gen_machine_topical_flan_t5_small", ("sci_gen", "flan_t5_small")),
    ],
)
def test_parse_mage_source(source, expected):
    assert datasets.parse_mage_source(source) == expected


def test_mage_label_one_means_human(tmp_path):
    (tmp_path / "train.csv").write_text(
        "text,label,src\nhuman words here,1,cmv_human\nmachine words here,0,cmv_machine_topical_7B\n",
        encoding="utf-8-sig",
    )
    for name in ("valid.csv", "test.csv"):
        (tmp_path / name).write_text("text,label,src\n", encoding="utf-8")
    rows = list(datasets.mage_records(tmp_path, {}))
    assert [(r.label, r.generator_name, r.domain) for r in rows] == [
        (REAL, REAL_GENERATOR, "cmv"),
        (FAKE, "7B", "cmv"),
    ]


def test_mage_continuations_share_a_group_with_their_human_text(tmp_path):
    opening = "the quick brown fox jumps over the lazy dog near the river"
    (tmp_path / "train.csv").write_text(
        f"text,label,src\n{opening} and it rested,1,cmv_human\n", encoding="utf-8"
    )
    (tmp_path / "test.csv").write_text(
        f"text,label,src\n{opening} then it ran away,0,cmv_machine_continuation_7B\n", encoding="utf-8"
    )
    (tmp_path / "valid.csv").write_text("text,label,src\n", encoding="utf-8")
    first, second = datasets.mage_records(tmp_path, {})
    assert first.group_id == second.group_id


def test_mage_refuses_a_label_that_contradicts_its_source(tmp_path):
    (tmp_path / "train.csv").write_text("text,label,src\nwords,0,cmv_human\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contradicts"):
        list(datasets.mage_records(tmp_path, {}))


def test_hc3_groups_answers_by_question(tmp_path):
    row = {"question": "Why?", "human_answers": ["h1", "h2"], "chatgpt_answers": ["c1"], "source": "finance"}
    (tmp_path / "all.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    rows = list(datasets.hc3_records(tmp_path, {}))
    assert len(rows) == 3
    assert len({item.group_id for item in rows}) == 1
    assert sorted(item.label for item in rows) == [REAL, REAL, FAKE]


def test_detector_bias_maps_kind_to_label(tmp_path):
    rows = [
        {"text": "essay", "domain": "TOEFL_real_91", "name": "Real TOEFL", "kind": "Human-Written", "model": "Human"},
        {"text": "polished", "domain": "TOEFL_gpt4polished_91", "name": "Fake", "kind": "AI-Generated", "model": "GPT4"},
    ]
    (tmp_path / "all_data.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / datasets.NATIVE_ESSAYS_FILE).write_text(json.dumps([{"document": "native essay"}]), encoding="utf-8")
    result = list(datasets.detector_bias_records(tmp_path, {}))
    assert [(r.label, r.generator_name, r.domain) for r in result] == [
        (REAL, REAL_GENERATOR, "TOEFL_real_91"),
        (FAKE, "gpt4", "TOEFL_gpt4polished_91"),
        (REAL, REAL_GENERATOR, "HewlettStudentEssay_real_88"),
    ]
