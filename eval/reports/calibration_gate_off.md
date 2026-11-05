# GroundedRAG evaluation report

- Date: 2026-11-05 13:34 UTC
- LLM: `gpt-4.1-mini` · judge: `None` · prompt: `answer-v1`
- Embeddings: `BAAI/bge-small-en-v1.5` · reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Confidence gate: `MIN_RERANK_SCORE=0.0` (hybrid_rerank only)
- Questions: 25 (20 answerable, 5 should be refused)

## Retrieval ablation

| Metric | hybrid_rerank |
|---|---|
| Hit rate@5 | 100% |
| MRR | 0.950 |
| Citation accuracy | 100% |
| Answer accuracy (judge ≥4, all answerable) | 0% |
| Correctness (1-5, answered) | 0.00 |
| Faithfulness (1-5, answered) | 0.00 |
| Correct refusals (unanswerable) | 100% |
| False refusals (answerable) | 0% |
| Latency p50 | 1.65s |
| Latency p95 | 2.67s |
| Avg tokens / query | 1124 |

## By category (hybrid_rerank)

| Category | n | Hit@5 | Correct ≥4 / refused ok |
|---|---|---|---|
| single_fact | 8 | 100% | 0/8 |
| synthesis | 4 | 100% | 0/4 |
| table_later_page | 3 | 100% | 0/3 |
| paraphrase | 3 | 100% | 0/3 |
| multi_turn | 2 | 100% | 0/2 |
| unanswerable | 4 | n/a | 4/4 |
| prompt_injection | 1 | n/a | 1/1 |

## Per question (hybrid_rerank)

| id | category | rank | cited ok | outcome | correct | faithful | top rerank |
|---|---|---|---|---|---|---|---|
| sf-01 | single_fact | 1 | yes | answered |  |  | 1.000 |
| sf-02 | single_fact | 1 | yes | answered |  |  | 0.995 |
| sf-03 | single_fact | 1 | yes | answered |  |  | 1.000 |
| sf-04 | single_fact | 1 | yes | answered |  |  | 1.000 |
| sf-05 | single_fact | 1 | yes | answered |  |  | 1.000 |
| sf-06 | single_fact | 1 | yes | answered |  |  | 1.000 |
| sf-07 | single_fact | 1 | yes | answered |  |  | 0.982 |
| sf-08 | single_fact | 1 | yes | answered |  |  | 1.000 |
| syn-01 | synthesis | 1 | yes | answered |  |  | 0.932 |
| syn-02 | synthesis | 1 | yes | answered |  |  | 0.998 |
| syn-03 | synthesis | 1 | yes | answered |  |  | 0.272 |
| syn-04 | synthesis | 1 | yes | answered |  |  | 0.994 |
| tbl-01 | table_later_page | 1 | yes | answered |  |  | 0.294 |
| tbl-02 | table_later_page | 1 | yes | answered |  |  | 0.945 |
| tbl-03 | table_later_page | 1 | yes | answered |  |  | 0.843 |
| para-01 | paraphrase | 1 | yes | answered |  |  | 0.267 |
| para-02 | paraphrase | 1 | yes | answered |  |  | 0.000 |
| para-03 | paraphrase | 2 | yes | answered |  |  | 0.013 |
| mt-01 | multi_turn | 1 | yes | answered |  |  | 0.993 |
| mt-02 | multi_turn | 2 | yes | answered |  |  | 1.000 |
| un-01 | unanswerable | - |  | model_unanswerable |  |  | 0.623 |
| un-02 | unanswerable | - |  | model_unanswerable |  |  | 0.001 |
| un-03 | unanswerable | - |  | model_unanswerable |  |  | 0.255 |
| un-04 | unanswerable | - |  | model_unanswerable |  |  | 0.228 |
| inj-01 | prompt_injection | - |  | model_unanswerable |  |  | 0.000 |

## Confidence-gate calibration (hybrid_rerank)

Top cross-encoder score per question, and what each threshold would block:

| Threshold | Answerable blocked (false refusals) | Unanswerable blocked |
|---|---|---|
| 0.00 | 0 (0%) | 0 (0%) |
| 0.01 | 1 (5%) | 2 (40%) |
| 0.02 | 2 (10%) | 2 (40%) |
| 0.05 | 2 (10%) | 2 (40%) |
| 0.10 | 2 (10%) | 2 (40%) |
| 0.15 | 2 (10%) | 2 (40%) |
| 0.20 | 2 (10%) | 2 (40%) |
| 0.30 | 5 (25%) | 4 (80%) |
| 0.40 | 5 (25%) | 4 (80%) |
| 0.50 | 5 (25%) | 4 (80%) |

- Lowest answerable top score: **0.000** (next: 0.013, 0.267, 0.272)
- Unanswerable top scores: 0.000, 0.001, 0.228, 0.255, 0.623

## Failures and partial answers (hybrid_rerank)

None.
