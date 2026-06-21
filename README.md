# dense-rubric

Evidence-attributed dense rubric RL: train **Qwen3.5-4B** on **HealthBench** where the rubric judge (**Gemini 3.5 Flash**) returns verbatim evidence quotes with character offsets, which become **per-token loss weights** in a CISPO policy-gradient loss.

The scalar weighted rubric score remains the only source of advantages. Validated evidence spans only *redistribute* per-token credit (mean weight renormalized to exactly 1.0).

## Layout

- `dense_rubric.ipynb` — the whole pipeline, blocked out in 12 labelled sections (Colab-ready, single GPU, no vLLM/DeepSpeed). Sections 0–7 run on CPU.
- `utils.py` — data loading, Gemini judge calls, span→token mapping, token weights, GDPO-decoupled advantages, DAPO overlong penalty, CISPO loss, HF generation, eval, wandb tables, checkpointing.
- `rubric/` — local fork of the [rubric](https://github.com/The-LLM-Data-Company/rubric) package (v2.3.0.dev0) with span support added: `EvidenceSpan`, `PerCriterionSpanOutput`, `SpanCriterionReport`, `span_validation.py`, `SpanPerCriterionGrader`.
- `smoke_test.py` — CPU-only end-to-end test (`python smoke_test.py`), no GPU or API keys needed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ./rubric
cp .env.example .env   # fill in GEMINI_API_KEY (and WANDB_API_KEY)
python smoke_test.py   # verify the pipeline on CPU
cd rubric && pytest    # 70 tests for the grader fork
```

On Colab: upload/clone the repo, add `GEMINI_API_KEY` to the Secrets sidebar, run the notebook top to bottom on an A100.

## Algorithm

CISPO with clip-higher (loss) + DAPO soft overlong punishment, no dynamic sampling (reward shaping) + GDPO decoupled advantage normalization for the two reward streams (rubric quality, length penalty). No KL term / reference model. See section 7 of the notebook for the math and the `DECOUPLED_NORM` flag for the GRPO-style A/B baseline.
