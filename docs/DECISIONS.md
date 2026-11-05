# Design decisions

Short ADR-style notes: what was decided, why, and what would change the decision.

---

### D1. One Postgres (pgvector) for vectors, full-text, sessions and logs
**Decision.** Store chunk embeddings (HNSW, cosine), the `tsvector` keyword index (GIN), chat
sessions, query logs and feedback in a single PostgreSQL 16 database.
**Why.** Hybrid search needs vectors and keyword ranking over the *same* rows; doing both in
one SQL engine avoids syncing two stores. Deleting a document cascades to its chunks in one
transaction, so a deleted document can never be retrieved again. A support knowledge base
is thousands to low millions of chunks, well inside pgvector/HNSW's comfortable range, and
the client runs one database instead of two.
**Revisit when.** More than about 10-50M vectors, very high QPS with strict p99 latency, or a need for
features like built-in multi-tenant sharding. Then Qdrant/Pinecone for vectors, keeping
Postgres for everything else.

### D2. Hybrid retrieval fused with Reciprocal Rank Fusion (k = 60)
**Decision.** Top-20 dense + top-20 full-text candidates, fused by RRF.
**Why.** Dense retrieval handles paraphrases ("card got declined" → "Failed payments");
keyword search handles exact identifiers that embeddings blur (`ORB-4091`,
`X-Orbitask-Signature`, plan names). RRF uses only ranks, so cosine similarity and
`ts_rank_cd` never need normalising onto one scale. k = 60 is the value from the original
paper and is robust; it was not tuned on the eval set, to avoid overfitting 25 questions.
The keyword query is an OR of the question's stemmed lexemes (not `plainto_tsquery`, which
ANDs every word and returns nothing for most full-sentence questions).

### D3. Cross-encoder reranking, scores squashed with a sigmoid
**Decision.** Rerank the fused candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2` and keep
the top 5. Raw logits pass through a sigmoid so scores live in [0, 1].
**Why.** A cross-encoder reads query and passage together, so it is far more precise than
bi-encoder similarity for the final ordering. The reranker sees `embed_text` (title +
section + text) because the section heading often carries the key term ("Refunds > Annual
plans"). The sigmoid only changes the scale; it makes the gate threshold readable.

### D4. The confidence gate is a cheap off-topic filter, not the main refusal mechanism
**Decision.** `MIN_RERANK_SCORE = 0.0001`. If the best rerank score is below it, skip the LLM
and return the fallback (`low_retrieval_confidence`).
**Why (from the calibration run, `eval/reports/calibration_gate_off.md`).** With the gate
disabled, top rerank scores were:
- clearly off-topic probes (weather, jokes, recipes) and the prompt-injection question:
  0.000013-0.000043;
- the *lowest* answerable question (a paraphrase, "the API keeps telling me I'm sending
  too much traffic"): 0.000373, about 10x higher; the next lowest was 0.0128;
- *plausible but unanswerable* questions: 0.0005, 0.23, 0.26, 0.62 ("is there a
  self-hosted version?"); an extra probe about the stock price reached 0.85.

So the score separates "about our product" from "not about our product" well, but it
**cannot** separate "answerable" from "plausible but not covered". Any threshold high
enough to catch those (≥ 0.63) would have refused about 40% of answerable questions. The
threshold sits between the off-topic cluster and the lowest answerable score (about 3x margin
each side). Plausible-but-unanswerable questions are handled by the next layers: the model's
structured `answerable=false`, then citation validation. With the gate off, those layers
alone refused all 5 should-refuse questions and none of the 20 answerable ones.
**Revisit when.** The knowledge base or the reranker changes. Re-run
`uv run python -m eval.run_eval --min-rerank-score 0 --no-judge` and read the calibration
table in the report.

### D5. Conversation history is used only to rewrite the query
**Decision.** The last `HISTORY_TURNS` exchanges feed a query-rewrite prompt that turns
"what about the annual plan?" into a standalone question. The answering prompt never sees
the history.
**Why.** If the answer prompt saw prior turns, the model could "answer" from its own earlier
(possibly wrong) replies instead of from documents. Rewriting keeps follow-ups working while
every answer stays grounded in freshly retrieved sources. If the rewrite fails, the raw
message is used.

### D6. Structured output, then validation, then a single fallback shape
**Decision.** The LLM must return `{answer, citations, answerable, confidence}` (OpenAI
`json_schema` strict mode; Anthropic forced tool call), validated with Pydantic. If
validation fails, retry once, then fall back with `llm_error`. Citations are checked against
the source ids actually sent. Invalid inline markers are stripped. An "answerable" answer
with no valid citation becomes `invalid_citations`.
**Why.** A free-text answer can't be checked automatically. Structured output gives the
service a machine-readable refusal signal and verifiable references. Every failure path
(`low_retrieval_confidence`, `model_unanswerable`, `invalid_citations`, `llm_error`) returns
the same HTTP 200 shape, so the client app needs one code path and can, for example, open a
ticket. An LLM outage never surfaces as a 5xx.

### D7. Prompt-injection defence in depth
**Decision.** Sources go inside `<source>` tags with `<` escaped (a document can't close its
own tag). The system prompt says source text is data, never instructions. Requests to ignore
the rules produce `answerable=false`. Off-topic prompts usually never reach the LLM (D4).
**Why.** No single layer is reliable against injection, but together they make the system
fail closed (fallback) rather than open.

### D8. Structure-aware chunking with contextual headers
**Decision.** Never let a chunk cross a heading boundary. Pack sentences up to about 500 tokens
with about 80 tokens of overlap inside a section. Keep tables whole (or split by rows with the
header repeated; tables are never overlapped). Embed and index
`title + "Section: <path>" + text`, but store and display the raw text.
**Why.** Section-bounded chunks make citations precise ("Refunds > Annual plans") and keep
facts with the heading that gives them meaning. The contextual header puts terms like the
plan or feature name into every chunk's embedding, even when the body says "it".

### D9. Approximate token counting
**Decision.** Count tokens as words + punctuation (regex) instead of a model tokenizer.
**Why.** It is deterministic, fast and tokenizer-agnostic (the embedder and the LLM use different
tokenizers anyway), and within about 15% of BPE counts for English. Chunk size is a soft target,
so exactness isn't needed. bge-small truncates at 512 wordpieces, and 500 "approx tokens"
stays under that for normal prose.

### D10. PDF parsing details
**Decision.** Headings come from font size relative to the body font. Running headers and
footers (lines repeated on ≥ 60% of pages, digits masked) are dropped. Tables are detected
with PyMuPDF and rebuilt cell by cell from word positions.
**Why.** `Table.to_markdown()` mistook the paragraph above the table for a header row, and
`Table.extract()` dropped spaces between words for Story-generated PDFs. Assigning words to
cells by centre point was the robust option. Scanned PDFs (no text layer) fail with a clear
"needs OCR" error instead of being silently ingested as empty.

### D11. Ingestion is idempotent by SHA-256 of the file bytes
**Decision.** Re-uploading identical bytes returns the existing document (HTTP 200). If that
earlier attempt *failed*, the upload is reprocessed (HTTP 202).
**Why.** The client can safely re-sync a whole folder. A failed parse is retried without
creating duplicates.

### D12. FastAPI BackgroundTasks instead of a job queue (for now)
**Decision.** Ingestion runs in-process after the upload response. Status is
`processing|ready|failed`, and failures store the error message.
**Why.** It is enough for a prototype and a small knowledge base, with no extra infrastructure. The
downside is that a process restart mid-ingestion leaves a document in `processing`. Next step: a
queue (Arq/Celery/SQS) with retries, and a sweeper that re-queues stale `processing` rows.

### D13. Embedding dimension is fixed when the schema is migrated
**Decision.** The pgvector column size comes from `EMBEDDING_DIM` at migration time, and
startup refuses to run if the configured embedder produces a different size.
**Why.** Mixing dimensions corrupts search silently. Switching to OpenAI embeddings means
a new migration and re-ingestion; that is a deliberate operation, not a config flip.

### D14. Models baked into the Docker image; CPU-only torch
**Decision.** The Dockerfile downloads both models at build time and sets `HF_HUB_OFFLINE=1`.
The torch wheel comes from the PyTorch CPU index on Linux.
**Why.** Startup is fast and deterministic and has no runtime dependency on the Hugging Face Hub. The
CPU wheel keeps the image about 1.5 GB smaller than the CUDA build.

### D15. Evaluation runs in-process through the same `ChatService` as the API
**Decision.** `eval/run_eval.py` builds the real components and calls `ChatService.handle`.
Ablations swap only the `Retriever` mode. The judge is `gpt-4.1` while the generator is
`gpt-4.1-mini`.
**Why.** It measures exactly the production code path (rewrite, gate, validation, fallback)
while still exposing internals (retrieved chunks, rerank scores) that the public API
deliberately hides. Using a different, stronger judge model reduces self-preference bias.
Eval sessions are deleted afterwards so they don't pollute `query_logs`.

### D16. Offline fakes ship in `app/testing.py`
**Decision.** Hash-based embedder, term-overlap reranker and a scripted LLM, selectable with
`EMBEDDING_PROVIDER=fake` / `LLM_PROVIDER=fake`.
**Why.** Tests run fast and offline, and with no API key needed. The same fakes let anyone boot the full
API without model downloads or keys for a demo of the plumbing.

### D17. Default ports 8000 (API) and 5433 (Postgres)
Port 5433 avoids clashing with a local Postgres on 5432. Both can be overridden with
`API_HOST_PORT` / `POSTGRES_HOST_PORT`.
