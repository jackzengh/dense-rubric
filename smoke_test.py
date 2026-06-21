"""CPU smoke test for the whole dense-rubric pipeline (no GPU, no API keys).

Mirrors the notebook's section-5 cells, then goes further: runs a complete
generate -> fake-grade -> advantages -> weights -> CISPO backward -> optimizer step
on a tiny random Qwen2 model. Run: python smoke_test.py
"""

import numpy as np
import torch
from transformers import AutoTokenizer

from rubric import EvaluationReport, EvidenceSpan, SpanCriterionReport, validate_spans
from utils import (
    build_token_weights,
    calculate_advantages_and_weights,
    compute_advantages,
    compute_cispo_loss,
    compute_token_log_probs,
    generate_completions,
    overlong_penalty,
    prepare_model_inputs,
    token_char_ranges,
)

MODEL_NAME = "Qwen/Qwen3.5-4B"


def test_span_validation():
    text = ("I understand this is scary. Call 911 or go to the ER right away. "
            "Chest pain can be serious. Do not drive yourself.")
    q1 = "Call 911 or go to the ER right away."
    s1 = text.find(q1)
    spans = [
        EvidenceSpan(quote=q1, start_char=s1, end_char=s1 + len(q1)),                 # exact
        EvidenceSpan(quote="Chest pain can be serious.", start_char=0, end_char=26),  # wrong offsets
        EvidenceSpan(quote="Take two aspirin and rest tonight.", start_char=0, end_char=35),  # fake
    ]
    valid, stats = validate_spans(text, spans)
    assert stats["n_exact_offset"] == 1
    assert stats["n_recovered_by_find"] == 1
    assert stats["n_dropped"] == 1
    assert all(text[s:e] for s, e in valid)
    print("[1/6] span validation OK")


def test_token_mapping(tokenizer):
    # unicode, dosage text, emoji, CJK — the BPE edge cases
    for sample in [
        "The naïve approach: take 100 mg/day of CoQ10 — it's naïve, but common.",
        "Stay hydrated 💧 and rest. 多喝水，好好休息。Fever > 38.5°C → see a doctor.",
    ]:
        ids = tokenizer(sample, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(ids, skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        ranges = token_char_ranges(ids, tokenizer)
        assert len(ranges) == len(ids)
        assert ranges[-1][1] == len(decoded), "ranges must cover the full decoded string"
        assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])), "ranges must be contiguous"

        sub = "100 mg/day" if "100" in sample else "see a doctor"
        s = decoded.find(sub)
        covering = [i for i, (cs, ce) in enumerate(ranges) if cs < s + len(sub) and ce > s]
        assert sub in tokenizer.decode([ids[i] for i in covering]), "substring round-trip failed"
    print("[2/6] char->token mapping OK (incl. unicode/emoji/CJK)")


def test_token_weights(tokenizer):
    completion = ("Drink plenty of fluids and rest. See a doctor within 24 hours if the "
                  "fever persists or gets worse. Avoid strenuous exercise this week.")
    ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    ranges = token_char_ranges(ids, tokenizer)
    quote = "See a doctor within 24 hours"
    qs = completion.find(quote)
    report = EvaluationReport(score=0.8, report=[
        SpanCriterionReport(weight=5.0, requirement="Advises seeing a doctor", verdict="MET",
                            reason="", valid_spans=[(qs, qs + len(quote))]),
        SpanCriterionReport(weight=2.0, requirement="Recommends hydration", verdict="MET",
                            reason="", valid_spans=[]),
        SpanCriterionReport(weight=-3.0, requirement="Recommends antibiotics", verdict="UNMET",
                            reason=""),
    ])

    w_pos, stats = build_token_weights(len(ids), ranges, report, advantage=1.0)
    assert abs(sum(w_pos) / len(w_pos) - 1.0) < 1e-9, "mean must be exactly 1.0"
    boosted = [i for i, (cs, ce) in enumerate(ranges) if cs < qs + len(quote) and ce > qs]
    assert all(w_pos[i] > w_pos[0] for i in boosted)
    assert stats["n_applicable_spans"] == 1

    # sign gating: negative advantage ignores MET-positive spans
    w_neg, _ = build_token_weights(len(ids), ranges, report, advantage=-1.0)
    assert w_neg == [1.0] * len(ids)
    # ... but boosts MET-negative (failure) spans
    fail_report = EvaluationReport(score=0.1, report=[
        SpanCriterionReport(weight=-3.0, requirement="Recommends antibiotics", verdict="MET",
                            reason="", valid_spans=[(0, 30)]),
    ])
    w_fail, _ = build_token_weights(len(ids), ranges, fail_report, advantage=-1.0)
    assert max(w_fail) > 1.0
    # degenerate cases -> all ones
    assert build_token_weights(len(ids), ranges, None, 1.0)[0] == [1.0] * len(ids)
    assert build_token_weights(0, [], report, 1.0)[0] == []
    print("[3/6] token-weight construction OK")


def test_advantages():
    assert overlong_penalty(100, False, 1024, 256) == 0.0
    assert overlong_penalty(896, False, 1024, 256) == -0.5
    assert overlong_penalty(1024, False, 1024, 256) == -1.0
    assert overlong_penalty(50, True, 1024, 256) == -1.0

    scores, pens = [0.9, 0.8, 0.9, 0.8], [0.0, -1.0, -1.0, 0.0]
    adv_sum = compute_advantages(scores, pens, 4, lambda_len=0.3, decoupled=False)
    adv_dec = compute_advantages(scores, pens, 4, lambda_len=0.3, decoupled=True)
    assert adv_sum[2] < adv_sum[3], "summed: length penalty should drown quality (the failure mode)"
    assert adv_dec[2] > adv_dec[3], "decoupled: quality signal must survive"
    assert np.allclose(compute_advantages([0.5] * 4, [0.0] * 4, 4), 0.0), "zero-variance guard"
    assert not np.isnan(adv_dec).any()
    print("[4/6] overlong penalty + decoupled advantages OK")


def make_tiny_model(vocab_size):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=vocab_size, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512,
    )
    return Qwen2ForCausalLM(config)


def test_loss_and_training_step(tokenizer):
    torch.manual_seed(0)
    model = make_tiny_model(len(tokenizer))
    device = torch.device("cpu")

    # synthetic episodes: 4 completions = 2 prompts x G=2
    query_ids = [[1, 2, 3], [1, 2, 3], [4, 5], [4, 5]]
    response_ids = [[10, 11, 12, 13], [14, 15], [16, 17, 18], [19]]
    advantages = [1.0, -1.0, 0.5, -0.5]
    weights = [[1.2, 0.9, 0.9, 1.0], [1.0, 1.0], [1.0, 1.0, 1.0], [1.0]]

    inputs = prepare_model_inputs(query_ids, response_ids, advantages, weights,
                                  device, tokenizer.pad_token_id or 0)
    B, T = inputs["input_ids"].shape
    assert all(inputs[k].shape == (B, T) for k in
               ("attention_mask", "labels", "labels_mask", "advantages", "token_weights"))

    with torch.no_grad():
        inputs["old_logps"] = compute_token_log_probs(model, inputs, temperature=1.0)
    total = inputs["labels_mask"][:, 1:].sum().item()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss, metrics = compute_cispo_loss(model, inputs, total, 1.0, 0.2, 0.4)
    assert torch.isfinite(loss)
    assert abs(metrics["ratio_mean"] - 1.0) < 1e-4, "first pass is on-policy: ratio == 1"
    assert metrics["clip_fraction"] == 0.0
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(grad_norm) and grad_norm > 0
    optimizer.step()

    # second (off-policy) pass: policy moved, ratio must drift from 1
    loss2, metrics2 = compute_cispo_loss(model, inputs, total, 1.0, 0.2, 0.4)
    assert torch.isfinite(loss2)
    assert metrics2["ratio_mean"] != 1.0
    print(f"[5/6] CISPO loss + optimizer step OK "
          f"(off-policy ratio drifted to {metrics2['ratio_mean']:.4f})")


def test_generation_and_glue(tokenizer):
    torch.manual_seed(0)
    model = make_tiny_model(len(tokenizer))
    prompts = [tokenizer("Hello there", add_special_tokens=False)["input_ids"],
               tokenizer("How are you today?", add_special_tokens=False)["input_ids"]]
    queries, responses, finish_reasons = generate_completions(
        model, tokenizer, prompts, group_size=2, max_new_tokens=8,
        temperature=1.0, gen_batch_size=2,
    )
    assert len(queries) == len(responses) == len(finish_reasons) == 4
    assert all(0 < len(r) <= 8 for r in responses)
    assert all(fr in ("stop", "length") for fr in finish_reasons)

    # fake judge reports (one failed grading -> None) through the full glue function
    reports = []
    for resp in responses[:3]:
        text = tokenizer.decode(resp, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
        reports.append(EvaluationReport(score=0.5, report=[
            SpanCriterionReport(weight=3.0, requirement="says anything", verdict="MET",
                                reason="", valid_spans=[(0, len(text))] if text else []),
        ]))
    reports.append(None)  # judge failure path

    rewards, advantages, token_weights, stats = calculate_advantages_and_weights(
        responses, finish_reasons, reports, tokenizer, group_size=2,
        max_response_tokens=8, overlong_buffer=4,
    )
    assert rewards[3] == 0.0, "judge failure -> reward 0"
    assert token_weights[3] == [1.0] * len(responses[3]), "judge failure -> uniform weights"
    assert len(advantages) == 4 and not np.isnan(advantages).any()
    assert all(len(w) == len(r) for w, r in zip(token_weights, responses))
    assert 0.0 <= stats["span/validity_rate"] <= 1.0
    print("[6/6] generation + end-to-end glue OK")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    test_span_validation()
    test_token_mapping(tokenizer)
    test_token_weights(tokenizer)
    test_advantages()
    test_loss_and_training_step(tokenizer)
    test_generation_and_glue(tokenizer)
    print("\nall smoke tests passed")
