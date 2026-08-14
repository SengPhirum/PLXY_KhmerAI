# Troubleshooting

Ordered by where you are: setting up, using, deploying, or looking at a bad
answer. For production incidents use `docs/operations_runbook.md`.

## Setup

**`make setup` fails on a package**
```bash
python3 --version                 # must be 3.11 or 3.12
rm -rf .venv && bash scripts/bootstrap.sh
```

**`ModuleNotFoundError` for a first-party package**
Run from the repository root, with the virtualenv active. `pytest` and the
`python -m` entry points both resolve the root automatically; a bare
`python some/script.py` may not.

**`make doctor` shows warnings on a fresh checkout**
Expected. No `.env`, no Ollama and no index are all normal before setup. Each
warning prints the command that fixes it.

**Khmer displays as boxes or dotted circles in the terminal**
A font problem, not a data problem. Install Noto Sans Khmer. Confirm the data is
fine with:
```bash
python -c "from preprocessing import normalize_text; s='ការធានារយៈពេល ២៤ ខែ។'; print(normalize_text(s) == s)"
```

## Khmer text

**Normalisation changed text that looked correct**
```python
from preprocessing.unicode_normalization import normalize_khmer

r = normalize_khmer(text)
print(r.changes)  # exactly which rules fired, and how many times
```
Every rule is individually toggleable. If a rule is wrong for your data, disable
it in `NormalizationConfig` — and please add the case to
`preprocessing/tests/fixtures.py::VALID_KHMER`, which is the regression corpus
for exactly this.

**Text is mojibake after PDF extraction**
Likely a legacy Khmer font (Limon, ABC-Zerk) encoding Khmer as Latin code points.
The loader flags it as `legacy_khmer_font_suspected` and validation marks the
document `needs_review`. Fix at the source: re-export the PDF as Unicode.

**Search does not match an obvious Khmer word**
Khmer has no spaces, so whitespace tokenisation fails. Check what the tokenizer
produced:
```python
from preprocessing import tokenize_for_search

print(tokenize_for_search("សេវាកម្មដឹកជញ្ជូន"))
```

## Retrieval

**"I don't have that information" for everything**
```bash
ls -l data/index/ACTIVE
python -m rag.reindex --list
cat data/manifests/company_validation.json | python3 -m json.tool | head -40
```
The usual cause is `nothing_retrievable`: documents defaulted to
`confidentiality: internal`. Add `confidentiality: public` to the ones customers
may see.

**A document exists but is never retrieved**
```bash
curl -s -X POST localhost:8000/v1/rag/search -H 'Content-Type: application/json' \
  -d '{"query":"<question>","top_k":10}' | python3 -m json.tool
```
Then check, in order: is it `active` and unexpired? Is it `public`? Did the
confidence gate withhold it (`confidence: low`)? Is the chunk too large?

**Retrieval finds the right product but the wrong document**
Chunk size. Run `python -m evaluation.evaluate_retrieval --sweep-chunking 400,600,800`.

## API

**503 `capacity_exceeded`**
Working as designed — the queue is full. If it is constant, either capacity is
too low or `KHMERAI_MAX_ACTIVE_GENERATIONS` exceeds `OLLAMA_NUM_PARALLEL`.

**429 with `Retry-After`**
Rate limited. Raise `KHMERAI_RATE_LIMIT_REQUESTS`/`_BURST` if the client is
legitimate.

**Streaming shows nothing until the end**
A buffering proxy. `proxy_buffering off` and `gzip off` on `/v1/chat/stream`; the
API already sends `X-Accel-Buffering: no`.

**401 on an admin endpoint with the right key**
Check the key is actually set: `grep KHMERAI_ADMIN_API_KEY .env`. An unset key
disables admin endpoints entirely and returns 503, not 401.

**The API will not start**
Configuration validation fails loudly at startup by design. See the table in
`docs/operations_runbook.md#api-down`.

## Model

**Answers in English instead of Khmer**
Check `prompt_version` in the response, and that the system prompt loaded. If the
model itself drifts to English, that is an SFT dataset problem — measure with
`evaluate_language.py`.

**A model number comes back transliterated (`ក្យូអិន-៤៥០០`)**
The output guard detects this as `corrupted_identifiers`. It is a training-data
issue: the code-switching slice must contain examples where the identifier
survives verbatim.

**The model repeats one phrase forever**
Raise `repeat_penalty` to 1.1. Confirm with
`evaluation.metrics.khmer_fluency`, which reports `repetition_loop`.

**Empty or truncated answers**
Raise `KHMERAI_MAX_OUTPUT_TOKENS`, and check the startup `chat.prompt_budget`
warning — the prompt may be crowding out the output reserve.

## Training

**Out of memory on Colab**
Use a smaller memory profile: `--memory-profile 24gb` or `16gb`.

**"only N% of tokens are supervised"**
The completion mask does not match the tokenizer's chat template. Compare
`training/chat_template.py` with the tokenizer's `chat_template` attribute.

**"no LoRA target modules discovered"**
The model uses different projection names. Inspect and set them explicitly:
```python
print(sorted({n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}))
```

**The pre-upload gate blocks training**
Correct behaviour: the dataset contains PII or credentials. Read
`data/manifests/pre_upload_report.json`, fix the findings, regenerate.

**The fine-tune seems to have done nothing**
Almost always a chat-template mismatch between training and serving. The
`TEMPLATE` block in the Modelfile must match what training rendered.

## Deployment

**`ollama create` cannot find the GGUF**
Run the merge and export first (`docs/training_guide.md`).

**Ollama restarts repeatedly**
Memory. Confirm `OLLAMA_MAX_LOADED_MODELS=1` and reduce
`OLLAMA_NUM_PARALLEL`.

**launchd service will not load**
```bash
plutil -lint /Library/LaunchDaemons/com.company.khmerai.api.plist
sudo launchctl print system/com.company.khmerai.api
```

## Getting help

Include: the output of `make doctor`, the versions from `curl -s
localhost:8000/health`, the relevant log lines (already redacted), and the exact
command you ran.
