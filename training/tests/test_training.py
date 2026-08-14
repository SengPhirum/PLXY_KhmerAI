"""Training-stack tests that run without a GPU.

Covers everything up to the point where torch is needed: configuration loading,
memory profiles, the completion mask, dataset validation, split determinism,
leakage prevention and the CPT/DPO acceptance rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preprocessing.schemas import SFTRecord
from training.chat_template import (
    IGNORE_INDEX,
    ChatMLTemplate,
    build_completion_mask,
    render_conversation,
    supervised_token_ratio,
)
from training.common import (
    MEMORY_PROFILES,
    TrainingConfig,
    assert_upload_approved,
    build_run_manifest,
    detect_hardware,
    resolve_memory_profile,
    set_seed,
)
from training.dataset_loader import (
    build_splits,
    conversation_text,
    describe_mixture,
    load_preference_pairs,
    split_records,
    validate_records,
)
from training.train_cpt import should_accept_cpt
from training.train_dpo import audit_pairs, should_accept_dpo


class FakeTokenizer:
    """Character-level tokenizer with a ChatML template - enough to test masking."""

    chat_template = "chatml"

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(c) % 5000 for c in text]}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        return ChatMLTemplate().render(messages, add_generation_prompt=add_generation_prompt)


def _record(
    user: str, assistant: str, *, intent: str = "warranty", rid: str = "r1", **meta: object
) -> SFTRecord:
    metadata = {"id": rid, "intent": intent, "language": "km"}
    metadata.update(meta)
    return SFTRecord.model_validate(
        {
            "messages": [
                {"role": "system", "content": "អ្នកគឺជាជំនួយការបម្រើអតិថិជន។"},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "metadata": metadata,
        }
    )


ANSWER = "ម៉ូដែល QN-4500A មានការធានារយៈពេល ២៤ ខែ ចាប់ពីថ្ងៃទិញ។"


# --- configuration ----------------------------------------------------------
@pytest.mark.parametrize(
    "path",
    [
        "configs/training/sft_9b.yaml",
        "configs/training/sft_4b.yaml",
        "configs/training/cpt.yaml",
        "configs/training/dpo.yaml",
    ],
)
def test_training_configs_load(path: str) -> None:
    config = TrainingConfig.from_yaml(path)
    assert config.base_model.startswith("Qwen/")
    assert config.seed == 20260814
    assert config.max_grad_norm == 1.0
    assert config.fingerprint()


def test_9b_and_4b_configs_differ_meaningfully() -> None:
    nine = TrainingConfig.from_yaml("configs/training/sft_9b.yaml")
    four = TrainingConfig.from_yaml("configs/training/sft_4b.yaml")
    assert nine.base_model != four.base_model
    assert nine.output_dir != four.output_dir
    assert nine.memory_profile != four.memory_profile


def test_all_four_memory_profiles_exist() -> None:
    assert set(MEMORY_PROFILES) == {"16gb", "24gb", "40gb", "80gb"}
    for profile in MEMORY_PROFILES.values():
        assert profile.effective_batch_size >= 8
        assert profile.max_seq_length >= 2048
    assert MEMORY_PROFILES["16gb"].load_in_4bit is True
    assert MEMORY_PROFILES["80gb"].load_in_4bit is False


def test_memory_profile_fills_defaults_but_respects_overrides() -> None:
    config = resolve_memory_profile(TrainingConfig(memory_profile="16gb"))
    assert config.per_device_train_batch_size == 1
    assert config.max_seq_length == 2048

    explicit = resolve_memory_profile(TrainingConfig(memory_profile="16gb", max_seq_length=1024))
    assert explicit.max_seq_length == 1024


def test_unknown_memory_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown memory profile"):
        resolve_memory_profile(TrainingConfig(memory_profile="999gb"))


def test_config_fingerprint_changes_with_hyperparameters() -> None:
    a = TrainingConfig(learning_rate=1e-4)
    b = TrainingConfig(learning_rate=2e-4)
    assert a.fingerprint() != b.fingerprint()


# --- reproducibility --------------------------------------------------------
def test_set_seed_is_safe_without_torch() -> None:
    set_seed(123)
    import random

    first = random.random()
    set_seed(123)
    assert random.random() == first


def test_run_manifest_records_everything_required(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps({"a": 1}) + "\n", encoding="utf-8")

    manifest = build_run_manifest(
        TrainingConfig(run_name="t"), stage="sft", dataset_paths={"train": str(dataset)}
    )
    for key in (
        "stage",
        "base_model",
        "model_revision",
        "datasets",
        "preprocessing_version",
        "code_commit",
        "seed",
        "hyperparameters",
        "environment",
        "hardware",
    ):
        assert key in manifest, f"run manifest is missing {key}"
    assert manifest["datasets"]["train"]["sha256"]
    assert manifest["hardware"]["python"]


def test_detect_hardware_works_without_torch() -> None:
    info = detect_hardware()
    assert info["platform"]
    assert "cpu_count" in info


# --- upload gate ------------------------------------------------------------
def test_upload_gate_blocks_a_missing_report(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no pre-upload scan"):
        assert_upload_approved(tmp_path / "missing.json")


def test_upload_gate_blocks_an_unapproved_report(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"approved": False, "blocking_findings": {"aws_access_key": 1}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="did NOT approve"):
        assert_upload_approved(report)


def test_upload_gate_passes_an_approved_report(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"approved": True, "blocking_findings": {}}), encoding="utf-8")
    assert_upload_approved(report)  # must not raise


# --- chat template / masking -----------------------------------------------
def test_render_uses_the_tokenizer_template() -> None:
    rendered = render_conversation(
        [{"role": "user", "content": "សួស្តី"}], FakeTokenizer(), system_prompt=None
    )
    assert "<|im_start|>user" in rendered
    assert "សួស្តី" in rendered


def test_generation_prompt_is_appended() -> None:
    rendered = render_conversation(
        [{"role": "user", "content": "សួស្តី"}],
        FakeTokenizer(),
        system_prompt=None,
        add_generation_prompt=True,
    )
    assert rendered.rstrip().endswith("<|im_start|>assistant")


def test_completion_mask_supervises_only_assistant_tokens() -> None:
    encoded = build_completion_mask(
        [
            {"role": "system", "content": "អ្នកគឺជាជំនួយការ"},
            {"role": "user", "content": "តើធានាប៉ុន្មានខែ?"},
            {"role": "assistant", "content": ANSWER},
        ],
        FakeTokenizer(),
        system_prompt=None,
    )
    assert len(encoded["input_ids"]) == len(encoded["labels"])
    assert IGNORE_INDEX in encoded["labels"]
    supervised = sum(1 for t in encoded["labels"] if t != IGNORE_INDEX)
    assert 0 < supervised < len(encoded["labels"])
    # The supervised span must be a suffix of the sequence for a single-turn record.
    first_supervised = next(i for i, t in enumerate(encoded["labels"]) if t != IGNORE_INDEX)
    assert all(t != IGNORE_INDEX for t in encoded["labels"][first_supervised:])


def test_completion_mask_handles_multiple_assistant_turns() -> None:
    encoded = build_completion_mask(
        [
            {"role": "user", "content": "សួស្តី"},
            {"role": "assistant", "content": "សួស្តី តើខ្ញុំអាចជួយអ្វី?"},
            {"role": "user", "content": "តើធានាប៉ុន្មានខែ?"},
            {"role": "assistant", "content": ANSWER},
        ],
        FakeTokenizer(),
        system_prompt=None,
    )
    labels = encoded["labels"]
    # Two separate supervised spans separated by masked prompt tokens.
    spans = 0
    previous_masked = True
    for token in labels:
        if token != IGNORE_INDEX and previous_masked:
            spans += 1
        previous_masked = token == IGNORE_INDEX
    assert spans == 2


def test_supervised_ratio_is_in_a_healthy_range() -> None:
    encoded = build_completion_mask(
        [
            {"role": "user", "content": "តើម៉ូដែល QN-4500A ធានាប៉ុន្មានខែ?"},
            {"role": "assistant", "content": ANSWER},
        ],
        FakeTokenizer(),
        system_prompt=None,
    )
    assert 0.1 < supervised_token_ratio(encoded) < 0.9


def test_truncation_keeps_the_final_assistant_turn() -> None:
    encoded = build_completion_mask(
        [
            {"role": "user", "content": "ក" * 500},
            {"role": "assistant", "content": ANSWER},
        ],
        FakeTokenizer(),
        system_prompt=None,
        max_length=100,
    )
    assert len(encoded["input_ids"]) == 100
    assert any(t != IGNORE_INDEX for t in encoded["labels"]), "truncation dropped the answer"


def test_empty_conversation() -> None:
    assert build_completion_mask([], FakeTokenizer(), system_prompt=None)["input_ids"] == []


# --- dataset validation -----------------------------------------------------
def test_sft_schema_rejects_malformed_records() -> None:
    with pytest.raises(ValueError, match="final message"):
        SFTRecord.model_validate(
            {
                "messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
                "metadata": {"id": "x"},
            }
        )
    with pytest.raises(ValueError, match="unknown intent"):
        SFTRecord.model_validate(
            {
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                ],
                "metadata": {"id": "x", "intent": "not_a_real_intent"},
            }
        )


def test_validate_removes_duplicates_and_low_quality() -> None:
    records = [
        _record("តើធានាប៉ុន្មានខែ?", ANSWER, rid="a"),
        _record("តើធានាប៉ុន្មានខែ?", ANSWER, rid="b"),  # exact duplicate
        _record("តើតម្លៃប៉ុន្មាន?", "ok", rid="c"),  # answer too short / not Khmer
    ]
    kept, stats = validate_records(records)
    assert stats.total == 3
    assert stats.exact_duplicates == 1
    assert stats.low_quality == 1
    assert len(kept) == 1


def test_validate_reports_intent_coverage() -> None:
    _, stats = validate_records([_record("តើធានាប៉ុន្មានខែ?", ANSWER)])
    coverage = stats.intent_coverage
    assert coverage["total"] > 20
    assert "missing" in coverage
    assert coverage["covered"] >= 1


def test_leakage_checker_blocks_a_test_record_from_training() -> None:
    from preprocessing.near_dedup import LeakageChecker, NearDedupConfig

    test_record = _record("តើធានាប៉ុន្មានខែ?", ANSWER, rid="test")
    checker = LeakageChecker(NearDedupConfig.for_sft())
    checker.protect("test", conversation_text(test_record))

    kept, stats = validate_records([test_record], leakage_checker=checker)
    assert stats.leaked == 1
    assert not kept


def test_split_is_deterministic_and_stable() -> None:
    records = [_record(f"សំណួរទី {i} អំពីការធានា?", ANSWER + f" ({i})", rid=str(i)) for i in range(60)]
    first = split_records(records)
    second = split_records(records)
    for name in first:
        assert [r.metadata.id for r in first[name]] == [r.metadata.id for r in second[name]]

    # Adding records must not move existing ones between splits.
    grown = split_records([*records, _record("សំណួរថ្មី?", ANSWER + " new", rid="new")])
    original_train = {r.metadata.id for r in first["train"]}
    grown_train = {r.metadata.id for r in grown["train"]}
    assert original_train <= grown_train


def test_adversarial_records_get_their_own_split() -> None:
    records = [
        _record("សួស្តី", ANSWER, intent="unsupported", rid="adv1"),
        _record("តើធានាប៉ុន្មានខែ?", ANSWER, intent="warranty", rid="ok1"),
    ]
    splits = split_records(records)
    assert [r.metadata.id for r in splits["adversarial"]] == ["adv1"]


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        split_records([], ratios={"train": 0.5, "test": 0.2})


def test_build_splits_writes_files_and_checks_leakage(tmp_path: Path) -> None:
    source = tmp_path / "sft.jsonl"
    records = [
        _record(f"សំណួរទី {i} អំពីការធានាផលិតផល?", ANSWER + f" លេខ {i}", rid=str(i)) for i in range(40)
    ]
    source.write_text("\n".join(r.model_dump_json() for r in records) + "\n", encoding="utf-8")

    stats = build_splits([source], tmp_path / "splits", report_path=tmp_path / "report.json")
    assert stats.valid > 0
    assert (tmp_path / "splits" / "train.jsonl").is_file()
    assert (tmp_path / "report.json").is_file()
    leakage = stats.intent_coverage["cross_split_leakage"]
    assert leakage["clean"], leakage


def test_describe_mixture_reports_the_delta_from_target() -> None:
    _, stats = validate_records(
        [
            _record("តើធានាប៉ុន្មានខែ?", ANSWER, intent="warranty", rid="a"),
            _record(
                "តើតម្លៃប៉ុន្មានដែរ?", "តម្លៃលក់រាយគឺ ៥២០ ដុល្លារ រួមបញ្ចូលពន្ធ។", intent="pricing", rid="b"
            ),
        ]
    )
    mixture = describe_mixture(stats)
    assert "target" in mixture and "achieved" in mixture and "delta" in mixture
    assert mixture["achieved"]["company_support"] > 0


# --- preference data --------------------------------------------------------
def test_preference_pairs_are_validated(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                {"prompt": "តើធានាប៉ុន្មានខែ?", "chosen": ANSWER, "rejected": "ប្រហែល ៦ ខែមើលទៅ។"},
                {"prompt": "incomplete", "chosen": "x"},
                {"prompt": "same", "chosen": "ដូចគ្នា", "rejected": "ដូចគ្នា"},
            )
        ),
        encoding="utf-8",
    )
    pairs = load_preference_pairs(path)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] != pairs[0]["rejected"]


def test_pair_audit_detects_length_bias() -> None:
    audit = audit_pairs(
        [{"prompt": f"q{i}", "chosen": "ចម្លើយវែងណាស់ " * 20, "rejected": "ខ្លី"} for i in range(10)]
    )
    assert audit["length_bias_warning"] is True
    assert audit["sufficient"] is False  # fewer than the 1000-pair minimum


# --- acceptance rules -------------------------------------------------------
def test_cpt_is_rejected_without_a_material_khmer_gain() -> None:
    accept, reasons = should_accept_cpt(
        {"khmer_perplexity": 10.0, "english_perplexity": 8.0},
        {"khmer_perplexity": 9.9, "english_perplexity": 8.0},
    )
    assert not accept
    assert any("below the" in r for r in reasons)


def test_cpt_is_rejected_when_english_regresses() -> None:
    accept, reasons = should_accept_cpt(
        {"khmer_perplexity": 10.0, "english_perplexity": 8.0},
        {"khmer_perplexity": 8.0, "english_perplexity": 9.0},
    )
    assert not accept
    assert any("English perplexity regressed" in r for r in reasons)


def test_cpt_is_accepted_on_a_clean_win() -> None:
    accept, reasons = should_accept_cpt(
        {"khmer_perplexity": 10.0, "english_perplexity": 8.0, "reasoning_accuracy": 0.70},
        {"khmer_perplexity": 8.5, "english_perplexity": 8.0, "reasoning_accuracy": 0.70},
    )
    assert accept, reasons


def test_dpo_is_rejected_when_hallucination_worsens() -> None:
    accept, reasons = should_accept_dpo(
        {"support_accuracy": 0.85, "hallucination_rate": 0.02},
        {"support_accuracy": 0.90, "hallucination_rate": 0.05},
    )
    assert not accept
    assert any("hallucination_rate" in r for r in reasons)


def test_dpo_is_rejected_without_a_measurable_gain() -> None:
    accept, _ = should_accept_dpo({"support_accuracy": 0.85}, {"support_accuracy": 0.851})
    assert not accept


def test_dpo_is_accepted_on_a_clean_win() -> None:
    accept, reasons = should_accept_dpo(
        {"support_accuracy": 0.85, "hallucination_rate": 0.02, "grounding_precision": 0.92},
        {"support_accuracy": 0.90, "hallucination_rate": 0.018, "grounding_precision": 0.93},
    )
    assert accept, reasons
