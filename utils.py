"""Logic for the evidence-attributed dense rubric RL notebook.

Everything here is plain PyTorch + HuggingFace — deliberately NO vllm and NO deepspeed
(single-GPU Colab; generation and training run sequentially on the same model).

Pipeline per iteration:
    generate G completions per prompt (HF generate, no grad)
      -> decode each completion ONCE; that exact string goes to the judge and all
         character offsets refer to it
      -> SpanPerCriterionGrader: scalar rubric score + validated evidence spans
      -> advantages from the scalar score (+ DAPO overlong penalty as a second,
         GDPO-style separately-normalized stream)
      -> evidence spans -> per-token loss WEIGHTS (mean 1.0, pure redistribution)
      -> CISPO loss with clip-higher: -clip(ratio) * A * logp * weight
"""

import asyncio
import json
import os
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

from rubric.autograders.schemas import PerCriterionOutput, PerCriterionSpanOutput
from rubric.types import Criterion

###############################################################################
# DATA: HealthBench download / split / tokenization
###############################################################################

HEALTHBENCH_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/"
    "2025-05-07-06-14-12_oss_eval.jsonl"
)


def load_healthbench_dataset(
    cache_dir: str | Path = "data",
    url: str = HEALTHBENCH_URL,
    split_frac: float = 0.9,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Download HealthBench (5,000 examples), shuffle, and split into train/val.

    Each returned example is a dict:
        {"prompt_id": str,
         "messages":  [{"role": ..., "content": ...}, ...],   # the conversation
         "rubrics":   [{"criterion": str, "points": int}, ...]}

    Examples whose rubric has no positive points are dropped: the HealthBench score
    is earned_points / total_POSITIVE_points, so they can't be scored.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "healthbench_oss_eval.jsonl"

    if not cache_path.exists():
        print(f"Downloading HealthBench to {cache_path} ...")
        with urllib.request.urlopen(url) as resp:
            cache_path.write_bytes(resp.read())

    examples = []
    with open(cache_path) as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            rubrics = [
                {"criterion": r["criterion"], "points": int(r["points"])}
                for r in raw["rubrics"]
            ]
            if sum(max(0, r["points"]) for r in rubrics) <= 0:
                continue  # unscorable: no positive points to normalize by
            examples.append(
                {
                    "prompt_id": raw["prompt_id"],
                    "messages": [
                        {"role": m["role"], "content": m["content"]} for m in raw["prompt"]
                    ],
                    "rubrics": rubrics,
                }
            )

    rng = random.Random(seed)
    rng.shuffle(examples)
    n_train = int(len(examples) * split_frac)
    train, val = examples[:n_train], examples[n_train:]
    print(f"HealthBench: {len(examples)} scorable examples -> {len(train)} train / {len(val)} val")
    return train, val


def make_criteria(rubrics: list[dict]) -> list[Criterion]:
    """HealthBench rubric items -> rubric-package Criterion objects.

    points becomes weight directly: positive points reward desired behavior,
    negative points are errors (the grader prompt branches on the sign).
    """
    return [Criterion(weight=float(r["points"]), requirement=r["criterion"]) for r in rubrics]


def prepare_examples(examples: list[dict], tokenizer, max_prompt_tokens: int = 1024) -> list[dict]:
    """Tokenize each example's conversation into prompt token ids.

    add_generation_prompt=True appends the assistant-turn prefix so the model knows
    it's its turn to speak. Also builds the plain-text `query` string the judge sees
    as conversation context. Prompts longer than max_prompt_tokens are dropped
    (they would crowd out the response within the context budget).
    """
    prepared, n_dropped = [], 0
    for ex in examples:
        kwargs = {"add_generation_prompt": True, "tokenize": True}
        try:
            # Qwen3-family templates accept enable_thinking; pass False so completions
            # are plain answers (otherwise evidence spans land inside <think> blocks).
            ids = tokenizer.apply_chat_template(ex["messages"], enable_thinking=False, **kwargs)
        except TypeError:
            ids = tokenizer.apply_chat_template(ex["messages"], **kwargs)
        if hasattr(ids, "keys"):  # some versions return a BatchEncoding dict
            ids = ids["input_ids"]
        if len(ids) > max_prompt_tokens:
            n_dropped += 1
            continue
        prepared.append(
            {
                **ex,
                "prompt_token_ids": list(ids),
                "query": "\n".join(f"{m['role']}: {m['content']}" for m in ex["messages"]),
            }
        )
    if n_dropped:
        print(f"prepare_examples: dropped {n_dropped} examples over {max_prompt_tokens} prompt tokens")
    return prepared


###############################################################################
# JUDGE: Gemini structured-output generate fns + batched grading
###############################################################################

DEFAULT_JUDGE_MODEL = "gemini-3.5-flash"

_GENAI_CLIENT = None  # lazy singleton so importing utils.py never requires an API key


async def _gemini_structured(system_prompt, user_prompt, schema, timeout_s, max_retries=4):
    """One structured-output Gemini call with timeout + exponential-backoff retries."""
    from google import genai
    from google.genai import errors, types

    global _GENAI_CLIENT
    if _GENAI_CLIENT is None:
        _GENAI_CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0,  # deterministic grading
        response_mime_type="application/json",
        response_schema=schema,
    )
    model = os.getenv("DR_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

    last_err = None
    for attempt in range(max_retries):
        try:
            async with asyncio.timeout(timeout_s):
                response = await _GENAI_CLIENT.aio.models.generate_content(
                    model=model, contents=user_prompt, config=config
                )
            if response.parsed is None:  # schema parse failure counts as an error
                raise ValueError(f"Gemini returned unparseable output: {response.text!r:.200}")
            return response.parsed
        except (TimeoutError, ConnectionError, ValueError, errors.APIError) as e:
            last_err = e
            if attempt < max_retries - 1:
                await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Gemini failed after {max_retries} attempts") from last_err


async def gemini_span_generate_fn(system_prompt: str, user_prompt: str, **kw) -> PerCriterionSpanOutput:
    """Judge call for training: verdict + evidence spans (longer JSON -> 30s timeout)."""
    timeout = float(os.getenv("DR_JUDGE_TIMEOUT_S", "30"))
    return await _gemini_structured(system_prompt, user_prompt, PerCriterionSpanOutput, timeout)


async def gemini_scalar_generate_fn(system_prompt: str, user_prompt: str, **kw) -> PerCriterionOutput:
    """Judge call for eval: verdict only (matches official HealthBench grading)."""
    timeout = float(os.getenv("DR_EVAL_JUDGE_TIMEOUT_S", "15"))
    return await _gemini_structured(system_prompt, user_prompt, PerCriterionOutput, timeout)


async def grade_completions(
    completions: list[str],
    criteria_lists: list[list[Criterion]],
    queries: list[str],
    grader,
    max_concurrent_completions: int = 8,
):
    """Grade every completion against its rubric, concurrently but bounded.

    The semaphore caps concurrent COMPLETIONS; each completion internally fires one
    judge call per criterion in parallel, so total in-flight Gemini calls is roughly
    max_concurrent_completions * criteria_per_example (~80 by default).

    A completion whose grading fails entirely returns None (caller treats it as
    reward 0.0 with uniform token weights) — one bad outlier beats crashing a whole
    rollout. Returns (reports, stats).
    """
    semaphore = asyncio.Semaphore(max_concurrent_completions)
    t_start = time.time()

    async def grade_one(completion, criteria, query):
        async with semaphore:
            try:
                return await grader.grade(completion, criteria, query=query)
            except Exception as e:
                print(f"grading failed: {type(e).__name__}: {e}")
                return None

    reports = await asyncio.gather(
        *[grade_one(c, cr, q) for c, cr, q in zip(completions, criteria_lists, queries)]
    )
    stats = {
        "judge/error_rate": sum(r is None for r in reports) / max(1, len(reports)),
        "judge/grading_seconds": time.time() - t_start,
    }
    return list(reports), stats


###############################################################################
# SPANS -> TOKENS: character offsets to token indices
###############################################################################


def token_char_ranges(token_ids: list[int], tokenizer) -> list[tuple[int, int]]:
    """Character range each token occupies in the decoded completion string.

    Uses incremental prefix decoding: ranges[i] = (len(decode(ids[:i])), len(decode(ids[:i+1]))).
    This is exactly consistent with the string the judge graded, BY CONSTRUCTION,
    because that string is decode(all ids) with the same flags. (Re-encoding the
    decoded text can produce different tokens than the ones actually sampled, and
    convert_ids_to_tokens breaks on byte-level BPE — both would misalign weights.)

    A token ending mid-UTF-8-codepoint decodes as a replacement char, so a boundary
    can be off by a character; the any-overlap rule in build_token_weights absorbs it.
    """
    decode_kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    ranges, prev_len = [], 0
    for i in range(1, len(token_ids) + 1):
        cur_len = len(tokenizer.decode(token_ids[:i], **decode_kwargs))
        ranges.append((prev_len, max(cur_len, prev_len)))
        prev_len = max(cur_len, prev_len)
    return ranges


def build_token_weights(
    response_len: int,
    char_ranges: list[tuple[int, int]],
    report,
    advantage: float,
    alpha: float = 0.5,
    max_boost: float = 1.0,
) -> tuple[list[float], dict]:
    """Turn validated evidence spans into per-token loss weights for ONE completion.

    The contract (user spec):
      - base weight 1.0 on every response token
      - advantage > 0: boost tokens inside spans of MET POSITIVE criteria
        (reinforce harder exactly where the good content lives)
      - advantage < 0: boost tokens inside spans of MET NEGATIVE criteria
        (suppress harder exactly where the error lives)
      - per-criterion boost alpha_c = alpha * |w_c| / sum(|w| over the whole rubric),
        so a criterion's boost is proportional to its share of rubric mass;
        overlapping criteria add up but are clipped at max_boost
      - finally renormalize so the MEAN weight over response tokens is exactly 1.0:
        the total gradient scale is unchanged vs the unweighted loss — spans only
        REDISTRIBUTE credit, they never add reward (scalar stays the source of truth)

    Degenerate cases (no report, zero advantage, no applicable spans, empty
    completion) return all-ones = the vanilla unweighted objective.
    """
    ones = [1.0] * response_len
    no_stats = {"fraction_tokens_boosted": 0.0, "max_token_weight": 1.0, "n_applicable_spans": 0}
    if response_len == 0 or report is None or report.report is None or advantage == 0.0:
        return ones, no_stats

    total_abs_weight = sum(abs(r.weight) for r in report.report)
    if total_abs_weight == 0:
        return ones, no_stats

    # pick which criteria's spans apply, by the sign of the advantage
    applicable = [
        r
        for r in report.report
        if getattr(r, "valid_spans", None)
        and r.verdict == "MET"
        and ((advantage > 0 and r.weight > 0) or (advantage < 0 and r.weight < 0))
    ]
    if not applicable:
        return ones, no_stats

    raw_boost = [0.0] * response_len
    n_spans = 0
    for r in applicable:
        alpha_c = alpha * abs(r.weight) / total_abs_weight
        for span_start, span_end in r.valid_spans:
            n_spans += 1
            for t, (cs, ce) in enumerate(char_ranges):
                if cs < span_end and ce > span_start:  # any-overlap
                    raw_boost[t] += alpha_c

    raw = [1.0 + min(max_boost, b) for b in raw_boost]
    scale = response_len / sum(raw)  # renormalize: mean weight = exactly 1.0
    weights = [w * scale for w in raw]

    stats = {
        "fraction_tokens_boosted": sum(b > 0 for b in raw_boost) / response_len,
        "max_token_weight": max(weights),
        "n_applicable_spans": n_spans,
    }
    return weights, stats


###############################################################################
# REWARD SHAPING + ADVANTAGES (DAPO soft overlong punishment, GDPO decoupled norm)
###############################################################################


def overlong_penalty(response_len: int, truncated: bool, max_len: int, buffer: int) -> float:
    """DAPO's soft overlong punishment (anti length-hacking reward stream).

    0 while comfortably short; ramps linearly to -1 inside the last `buffer` tokens
    before max_len; truncated (no EOS) responses get the full -1 — their reward is
    unreliable AND they hit the wall.
    """
    if truncated:
        return -1.0
    threshold = max_len - buffer
    if response_len <= threshold:
        return 0.0
    return -(response_len - threshold) / buffer


def _group_znorm(values: np.ndarray, group_size: int, eps: float = 1e-6) -> np.ndarray:
    """Normalize within each group of `group_size` completions (same prompt).

    Zero-variance guard: a group where every completion got the same value carries
    no preference signal — its advantages become 0 instead of noise (or NaN).
    """
    grouped = values.reshape(-1, group_size)
    std = grouped.std(axis=1, keepdims=True)
    normed = (grouped - grouped.mean(axis=1, keepdims=True)) / (std + eps)
    normed[np.broadcast_to(std < eps, normed.shape)] = 0.0
    return normed.reshape(-1)


def compute_advantages(
    rubric_scores: list[float],
    overlong_penalties: list[float],
    group_size: int,
    lambda_len: float = 0.3,
    decoupled: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """Combine the two reward streams (rubric quality, overlong penalty) into advantages.

    decoupled=True  (GDPO, arXiv:2601.05242): group-normalize EACH stream separately,
        weighted-sum them, then batch-normalize the combined advantage. Summing raw
        rewards first (GRPO-style) can collapse distinct (quality, length) combinations
        into identical advantages, losing signal resolution.
    decoupled=False (DAPO-style baseline for A/B comparison): add the penalty to the
        raw reward, then a single group normalization.
    """
    scores = np.asarray(rubric_scores, dtype=np.float64)
    penalties = np.asarray(overlong_penalties, dtype=np.float64)

    if decoupled:
        combined = _group_znorm(scores, group_size) + lambda_len * _group_znorm(
            penalties, group_size
        )
        std = combined.std()
        if std < eps:
            return np.zeros_like(combined)
        return (combined - combined.mean()) / (std + eps)  # batch-wise norm (GDPO)

    return _group_znorm(scores + lambda_len * penalties, group_size)


def calculate_advantages_and_weights(
    response_token_ids: list[list[int]],
    finish_reasons: list[str],
    reports: list,
    tokenizer,
    group_size: int,
    max_response_tokens: int,
    overlong_buffer: int = 256,
    lambda_len: float = 0.3,
    alpha: float = 0.5,
    max_boost: float = 1.0,
    decoupled_norm: bool = True,
):
    """Glue: grading reports -> rewards -> advantages -> per-token weights.

    Returns (rewards, advantages, token_weights, stats):
        rewards:       list[float], the scalar HealthBench scores (judge failure -> 0.0)
        advantages:    np.ndarray, one per completion
        token_weights: list[list[float]], aligned with response_token_ids
        stats:         aggregated wandb metrics
    """
    rewards = [r.score if r is not None else 0.0 for r in reports]
    penalties = [
        overlong_penalty(len(ids), fr != "stop", max_response_tokens, overlong_buffer)
        for ids, fr in zip(response_token_ids, finish_reasons)
    ]
    advantages = compute_advantages(
        rewards, penalties, group_size, lambda_len=lambda_len, decoupled=decoupled_norm
    )

    token_weights, weight_stats, span_counters = [], [], {}
    for ids, report, adv in zip(response_token_ids, reports, advantages):
        ranges = token_char_ranges(ids, tokenizer)
        weights, w_stats = build_token_weights(
            len(ids), ranges, report, float(adv), alpha=alpha, max_boost=max_boost
        )
        token_weights.append(weights)
        weight_stats.append(w_stats)
        if report is not None and report.report is not None:
            for cr in report.report:
                for key, value in getattr(cr, "span_stats", {}).items():
                    span_counters[key] = span_counters.get(key, 0) + value

    n_returned = max(1, span_counters.get("n_returned", 0))
    n_valid = n_returned - span_counters.get("n_dropped", 0)
    stats = {
        "train/reward": float(np.mean(rewards)),
        "train/overlong_penalty_mean": float(np.mean(penalties)),
        "train/advantage_std": float(np.std(advantages)),
        "train/non_stop_rate": float(np.mean([fr != "stop" for fr in finish_reasons])),
        "train/response_length": float(np.mean([len(ids) for ids in response_token_ids])),
        "span/validity_rate": n_valid / n_returned,
        "span/exact_offset_rate": span_counters.get("n_exact_offset", 0) / n_returned,
        "span/recovered_by_find_rate": span_counters.get("n_recovered_by_find", 0) / n_returned,
        "span/fraction_tokens_boosted": float(
            np.mean([s["fraction_tokens_boosted"] for s in weight_stats])
        ),
        "span/mean_max_token_weight": float(
            np.mean([s["max_token_weight"] for s in weight_stats])
        ),
    }
    return rewards, advantages, token_weights, stats


###############################################################################
# MODEL INPUTS + CISPO LOSS
###############################################################################


def prepare_model_inputs(
    query_token_ids: list[list[int]],
    response_token_ids: list[list[int]],
    advantages: list[float],
    token_weights: list[list[float]],
    device: torch.device,
    pad_token_id: int = 0,
):
    """Pad sequences and build every tensor the loss needs.

    Per sequence (then right-padded to the batch max):
        input_ids:     query + response + pad
        attention_mask: 1 on real tokens, 0 on pad
        labels:        -100 on query/pad (never train to predict the prompt), response ids elsewhere
        labels_mask:   1 only on response tokens (the loss is averaged over these)
        advantages:    the completion's scalar advantage broadcast over its response tokens
        token_weights: the evidence-derived per-token weights over response tokens
    """
    max_seq_len = max(len(q) + len(r) for q, r in zip(query_token_ids, response_token_ids))
    inputs = {
        "input_ids": [], "attention_mask": [], "labels": [], "labels_mask": [],
        "advantages": [], "token_weights": [],
    }
    for query, response, advantage, weights in zip(
        query_token_ids, response_token_ids, advantages, token_weights
    ):
        seq_len = len(query) + len(response)
        pad_len = max_seq_len - seq_len
        inputs["input_ids"].append(query + response + [pad_token_id] * pad_len)
        inputs["attention_mask"].append([1] * seq_len + [0] * pad_len)
        inputs["labels"].append([-100] * len(query) + response + [-100] * pad_len)
        inputs["labels_mask"].append([0] * len(query) + [1] * len(response) + [0] * pad_len)
        inputs["advantages"].append([0.0] * len(query) + [float(advantage)] * len(response) + [0.0] * pad_len)
        inputs["token_weights"].append([0.0] * len(query) + list(weights) + [0.0] * pad_len)

    return {
        k: torch.tensor(
            v, dtype=torch.float if k in ("advantages", "token_weights") else torch.long,
            device=device,
        )
        for k, v in inputs.items()
    }


def compute_token_log_probs(model, inputs: dict, temperature: float) -> torch.Tensor:
    """Log-prob the model assigns to each actual next token. Shape (B, T-1).

    Standard next-token shift: logits[:, :-1] predict labels[:, 1:]. Masked positions
    are zeroed (their label is -100, replaced with 0 before gather to avoid indexing
    errors — the mask zeroes them out afterwards anyway).
    """
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        return_dict=True,
        use_cache=False,
    )
    logits = outputs.logits.float() / temperature
    shift_logits = logits[..., :-1, :]
    shift_labels = inputs["labels"][..., 1:].clone()
    shift_mask = inputs["labels_mask"][..., 1:]
    shift_labels[~shift_mask.bool()] = 0
    log_probs = torch.gather(
        shift_logits.log_softmax(dim=-1), dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    return log_probs * shift_mask


def compute_cispo_loss(
    model,
    inputs: dict,
    total_response_tokens: int,
    temperature: float = 1.0,
    eps_low: float = 0.2,
    eps_high: float = 0.4,
):
    """CISPO policy-gradient loss with clip-higher and evidence token weights.

        ratio     = exp(logp - old_logp)               importance ratio vs the policy
                                                        that generated the rollout
        coef      = clip(ratio, 1-eps_low, 1+eps_high) * advantage   (DETACHED)
        per_token = -coef * logp * token_weight

    CISPO clips the RATIO inside a detached coefficient instead of clipping the
    update like PPO, so every token keeps a gradient (-coef * dlogp); clipping only
    caps how strongly off-policy tokens count. eps_high > eps_low (clip-higher,
    DAPO) lets low-probability tokens grow more before capping, fighting entropy
    collapse. token_weight redistributes credit toward validated evidence spans
    (mean 1.0 per completion, so the overall scale matches the unweighted loss).

    inputs must contain "old_logps" (B, T-1), computed once per iteration with
    torch.no_grad right after generation. The loss is summed and divided by the
    TOTAL response tokens in the whole iteration batch so gradient accumulation
    across microbatches reproduces the full-batch loss exactly.
    """
    logps = compute_token_log_probs(model, inputs, temperature)
    old_logps = inputs["old_logps"]
    mask = inputs["labels_mask"][..., 1:].float()

    ratio = torch.exp(logps - old_logps)
    clipped_ratio = torch.clamp(ratio, min=1 - eps_low, max=1 + eps_high)
    coef = (clipped_ratio * inputs["advantages"][..., 1:]).detach()

    per_token = -coef * logps * inputs["token_weights"][..., 1:]
    loss = (per_token * mask).sum() / total_response_tokens

    with torch.no_grad():
        clip_fraction = ((ratio != clipped_ratio).float() * mask).sum() / mask.sum().clamp(min=1)
        metrics = {
            "loss": loss.item(),
            "ratio_mean": ((ratio * mask).sum() / mask.sum().clamp(min=1)).item(),
            "clip_fraction": clip_fraction.item(),
            "policy_entropy_proxy": (-(logps * mask).sum() / mask.sum().clamp(min=1)).item(),
        }
    return loss, metrics


###############################################################################
# GENERATION (HF generate — sequential with training, no vLLM)
###############################################################################


@torch.no_grad()
def generate_completions(
    model,
    tokenizer,
    prompt_token_ids: list[list[int]],
    group_size: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    gen_batch_size: int = 16,
):
    """Sample group_size completions per prompt with HF generate, in chunks.

    Chunking (gen_batch_size sequences at a time) bounds the KV-cache memory, which
    is what blows up first on a single GPU that also holds optimizer state.

    Returns (query_ids, response_ids, finish_reasons), all flat lists of length
    len(prompt_token_ids) * group_size, grouped per prompt (prompt 0's G completions
    first, then prompt 1's, ...). EOS is KEPT in response ids — the model must learn
    to stop. finish_reason is "stop" if EOS was generated, else "length".
    """
    was_training = model.training
    model.eval()
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    flat_prompts = [ids for ids in prompt_token_ids for _ in range(group_size)]
    all_responses, all_finish_reasons = [], []

    for chunk_start in range(0, len(flat_prompts), gen_batch_size):
        chunk = flat_prompts[chunk_start : chunk_start + gen_batch_size]
        max_prompt_len = max(len(p) for p in chunk)
        # left-pad so every prompt ends at the same position and generation starts
        # in lockstep (right padding would put pad tokens inside the context)
        input_ids = torch.tensor(
            [[pad_id] * (max_prompt_len - len(p)) + p for p in chunk], device=model.device
        )
        attention_mask = torch.tensor(
            [[0] * (max_prompt_len - len(p)) + [1] * len(p) for p in chunk], device=model.device
        )
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        for row in out[:, max_prompt_len:].tolist():
            if eos_id in row:
                response = row[: row.index(eos_id) + 1]  # keep the EOS itself
                all_finish_reasons.append("stop")
            else:
                response = row
                all_finish_reasons.append("length")
            all_responses.append(response)

    if was_training:
        model.train()
    return flat_prompts, all_responses, all_finish_reasons


###############################################################################
# EVALUATION (every N iterations, official HealthBench scoring, scalar only)
###############################################################################


async def evaluate_on_validation(
    model,
    tokenizer,
    val_examples: list[dict],
    eval_grader,
    num_eval_samples: int = 64,
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
    gen_batch_size: int = 16,
    max_concurrent_completions: int = 8,
    seed: int = 0,
):
    """Generate one answer per validation prompt and grade with the PLAIN scalar
    grader (no spans): cheaper, no span-extraction failure modes in the metric, and
    identical to official HealthBench scoring (earned / total positive points).

    The subsample is FIXED by the seed so every eval scores the same prompts —
    changes in eval/healthbench_score reflect the model, not the sample.
    """
    rng = random.Random(seed)
    samples = rng.sample(val_examples, min(num_eval_samples, len(val_examples)))

    _, response_ids, finish_reasons = generate_completions(
        model, tokenizer, [s["prompt_token_ids"] for s in samples],
        group_size=1, max_new_tokens=max_new_tokens, temperature=temperature,
        gen_batch_size=gen_batch_size,
    )
    completions = [
        tokenizer.decode(ids, skip_special_tokens=True) for ids in response_ids
    ]
    reports, judge_stats = await grade_completions(
        completions,
        [make_criteria(s["rubrics"]) for s in samples],
        [s["query"] for s in samples],
        eval_grader,
        max_concurrent_completions=max_concurrent_completions,
    )
    scores = [r.score for r in reports if r is not None]
    metrics = {
        "eval/healthbench_score": float(np.mean(scores)) if scores else 0.0,
        "eval/non_stop_rate": float(np.mean([fr != "stop" for fr in finish_reasons])),
        "eval/response_length": float(np.mean([len(ids) for ids in response_ids])),
        "eval/judge_error_rate": judge_stats["judge/error_rate"],
    }
    rows = [
        {"prompt_id": s["prompt_id"], "completion": c,
         "score": r.score if r is not None else None}
        for s, c, r in zip(samples, completions, reports)
    ]
    return metrics, rows


###############################################################################
# WANDB TABLES: every criterion verdict + every evidence span, every iteration
###############################################################################


def log_criteria_and_spans_table(
    wandb_run,
    iteration: int,
    prompt_ids: list[str],
    completions: list[str],
    reports: list,
    rewards: list[float],
    advantages,
):
    """Log two tables to wandb:

    tables/completions    — one row per completion (prompt_id, text, reward, advantage)
    tables/criteria_spans — one row per (completion, criterion, span). Criteria with
        no surviving spans still get one row (quote empty) so every VERDICT is
        recorded, not just the localized ones.
    """
    import wandb

    completions_table = wandb.Table(
        columns=["iteration", "prompt_id", "completion_idx", "completion", "reward", "advantage"]
    )
    spans_table = wandb.Table(
        columns=[
            "iteration", "prompt_id", "completion_idx", "criterion", "weight", "verdict",
            "localizable", "n_spans_returned", "n_spans_valid", "quote", "start_char",
            "end_char", "reward", "advantage",
        ]
    )

    for idx, (pid, completion, report, reward, adv) in enumerate(
        zip(prompt_ids, completions, reports, rewards, advantages)
    ):
        completions_table.add_data(iteration, pid, idx, completion, reward, float(adv))
        if report is None or report.report is None:
            continue
        for cr in report.report:
            valid_spans = getattr(cr, "valid_spans", []) or []
            stats = getattr(cr, "span_stats", {}) or {}
            n_returned = stats.get("n_returned", 0)
            common = (
                iteration, pid, idx, cr.requirement, cr.weight, cr.verdict,
                getattr(cr, "localizable", True), n_returned, len(valid_spans),
            )
            if valid_spans:
                for start, end in valid_spans:
                    spans_table.add_data(*common, completion[start:end], start, end,
                                         reward, float(adv))
            else:
                spans_table.add_data(*common, "", -1, -1, reward, float(adv))

    wandb_run.log(
        {"tables/completions": completions_table, "tables/criteria_spans": spans_table},
        step=iteration,
    )


###############################################################################
# CHECKPOINTING (plain save_pretrained — no DeepSpeed)
###############################################################################


def save_checkpoint(model, optimizer, iteration: int, exp_dir: str | Path):
    ckpt_dir = Path(exp_dir) / "checkpoints" / f"ckpt_{iteration:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir / "model")
    torch.save(
        {"optimizer": optimizer.state_dict(), "iteration": iteration},
        ckpt_dir / "trainer_state.pt",
    )
    print(f"saved checkpoint -> {ckpt_dir}")


def find_last_checkpoint(exp_dir: str | Path):
    """Returns (path, iteration) of the newest complete checkpoint, or (None, None)."""
    checkpoint_dir = Path(exp_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        return None, None
    checkpoints = [
        c for c in checkpoint_dir.glob("ckpt_*") if (c / "trainer_state.pt").exists()
    ]
    if not checkpoints:
        return None, None
    last = max(checkpoints, key=lambda c: int(c.stem.split("_")[-1]))
    return last, int(last.stem.split("_")[-1])


def load_checkpoint(ckpt_path: Path, model, optimizer) -> int:
    """Load weights + optimizer state in place; returns the iteration to resume from."""
    from transformers import AutoModelForCausalLM

    restored = AutoModelForCausalLM.from_pretrained(ckpt_path / "model", torch_dtype=model.dtype)
    model.load_state_dict(restored.state_dict())
    del restored
    state = torch.load(ckpt_path / "trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    return state["iteration"]
