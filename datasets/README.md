# Datasets

Acquisition, provenance and licensing for the public Khmer corpora (Phase 2).

## Rules

1. **No data without a manifest.** Every file under `data/` must have a
   corresponding `data/manifests/<name>.json` recording where it came from, which
   revision, its licence, and its commercial-review status.
2. **No unreviewed data in production training.** Sources are `approved`,
   `review_required`, `prohibited_for_commercial` or `evaluation_only`. Only
   `approved` may train a production model.
3. **Evaluation data never trains.** Belebele and FLORES-Plus go to
   `data/evaluation/public/`, and `build_manifest.py --check` fails if such a
   file appears under a training path.

## Commands

```bash
python datasets/download_public.py --list
python datasets/download_public.py --all --dry-run        # no network
python datasets/download_public.py --all --limit 50000
python datasets/download_public.py --name khmer_wikipedia
python datasets/build_manifest.py --root data/raw
python datasets/build_manifest.py --root data/raw --check  # CI gate
python datasets/license_report.py
python datasets/license_report.py --fail-on-unreviewed     # release gate
```

## Sources

Configured in `configs/base.yaml -> datasets`.

| Source | Subset | Use | Status |
|---|---|---|---|
| `HuggingFaceFW/fineweb-2` | `khm_Khmr` | continued pretraining | review_required |
| `wikimedia/wikipedia` | `20231101.km` | continued pretraining | approved |
| `uonlp/CulturaX` | `km` | continued pretraining | review_required |
| `CohereForAI/aya_collection_language_split` | `khmer` | instruction tuning | review_required |
| `kimleang123/khmer_question_answer` | default | instruction tuning | review_required |
| `Helsinki-NLP/opus-100` | `en-km` | code switching | review_required |
| `facebook/belebele` | `khm_Khmr` | **evaluation only** | evaluation_only |
| `openlanguagedata/flores_plus` | `khm_Khmr` | **evaluation only** | evaluation_only |

Change a status in `configs/base.yaml` after legal review, then re-run
`license_report.py` to regenerate `docs/dataset_provenance.md`.

## Manifest schema

```json
{
  "dataset": "khmer_wikipedia",
  "source": "wikimedia/wikipedia",
  "revision": "20231101.km",
  "download_date": "2026-08-14T10:00:00Z",
  "license": "CC BY-SA 4.0 / GFDL",
  "language": "km",
  "subset": "20231101.km",
  "raw_records": 12345,
  "raw_bytes": 98765432,
  "sha256": "…",
  "intended_use": "continued_pretraining",
  "commercial_review_status": "approved"
}
```

## A note on the directory name

This directory is called `datasets/`, which is also the name of the Hugging Face
package. The scripts here add the repository root to `sys.path` explicitly and
import HF `datasets` lazily, so both resolve correctly. There is no
`__init__.py` here on purpose — it is a scripts directory, not a package.

## Pinning revisions

For a reproducible training run, pin the revision:

```yaml
- name: khmer_wikipedia
  hf_id: wikimedia/wikipedia
  subset: 20231101.km
  revision: <commit-sha>
```

An unpinned revision means the corpus can change under you and a "reproducible"
run silently is not.
