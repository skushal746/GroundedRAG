# GroundedRAG

**A customer-support assistant that answers only from your documents, cites every claim, and
says "I don't know" when it should.**

GroundedRAG turns product docs, FAQs, policies and API guides (PDF, DOCX, Markdown, HTML,
TXT) into a question-answering API that your existing app can call. The emphasis is on
*reliability*, not just wiring an LLM to a vector database:
- hybrid retrieval, then a cross-encoder reranker;
- a confidence gate before the LLM;
- structured output whose citations are validated against the sources actually provided;
- one consistent fallback (with a machine-readable reason) whenever an answer can't be
  trusted;
- an evaluation harness that measures all of this on a 25-question test set, including
  unanswerable and prompt-injection questions.

```mermaid
flowchart LR
    subgraph Ingestion["Ingestion (background task)"]
        U[POST /v1/documents] --> P[Parse<br/>PDF · DOCX · MD · HTML · TXT]
        P --> C[Structure-aware chunking<br/>+ contextual headers]
        C --> E[Embed<br/>bge-small-en-v1.5]
        E --> DB[(PostgreSQL 16<br/>pgvector HNSW + tsvector GIN<br/>sessions · query_logs · feedback)]
    end
    subgraph Chat["POST /v1/chat"]
        Q[Message + session] --> RW[Query rewrite<br/>follow-ups → standalone]
        RW --> V[Vector top-20]
        RW --> K[Full-text top-20]
        V --> F[RRF fusion]
        K --> F
        F --> RR[Cross-encoder rerank<br/>→ top-5, de-dupe]
        RR --> G{Confidence<br/>gate}
        G -- low --> FB[Fallback +<br/>fallback_reason]
        G -- ok --> L[LLM: structured JSON<br/>answer · citations · answerable]
        L --> CV{Citation<br/>validation}
        CV -- invalid / unanswerable / error --> FB
        CV -- grounded --> A[Answer + sources]
    end
    DB -. retrieval .-> V
    DB -. retrieval .-> K
```

---

## 1. Quickstart

Requirements: Docker, [uv](https://docs.astral.sh/uv/), and an OpenAI or Anthropic API key,
**or** a local [Ollama](https://ollama.com) with the `Qwen3-Coder` model (`ollama pull qwen3-coder`).
If the key for the configured provider is empty, GroundedRAG falls back to Ollama automatically
and logs `llm_fallback_to_ollama` at startup.

```bash
cp .env.example .env              # then set OPENAI_API_KEY (and change API_KEYS)
make up                           # builds the image (models baked in), starts api + postgres, waits for /ready
make ingest-samples               # uploads the 5 sample docs and waits until every one is "ready"
```

Ask a question:

```bash
curl -s localhost:8000/v1/chat \
  -H 'X-API-Key: dev-key-change-me' -H 'Content-Type: application/json' \
  -d '{"message": "How long do I have to request a refund on an annual plan?"}'
```

```json
{
  "session_id": "76a5b6c9-9e38-4707-83b7-ae757a37e8b1",
  "message_id": "d6b4f2b8-87f0-40ff-9593-6501a570b186",
  "answer": "You have 30 days from the initial purchase or annual renewal to request a full refund on an annual plan. This refund window applies to purchases and renewals made on or after 1 March 2026. After 30 days, annual subscriptions are non-refundable, but you can turn off auto-renewal at any time [S1] [S3].",
  "answerable": true,
  "confidence": "high",
  "fallback_reason": null,
  "sources": [
    {"id": "S1", "filename": "orbitask_billing_and_refunds.docx", "page": null,
     "section": "Refunds > Annual plans", "score": 0.9998, "snippet": "Annual subscriptions are eligible for a full refund if you request it within 30 days..."},
    {"id": "S3", "filename": "orbitask_release_notes.txt", "page": null,
     "section": "Version 4.3 (1 March 2026)", "score": 0.9981, "snippet": "Billing: the refund window for annual plans has been extended from 14 days to 30 days..."}
  ],
  "latency_ms": 2868
}
```

Follow up in the same session: `{"session_id": "<from above>", "message": "And what about monthly plans?"}`.
It is rewritten to *"What is the refund policy for monthly plans?"* before retrieval. Ask
something the docs don't cover (e.g. *"What uptime SLA do you guarantee?"*) and you get the
fallback with `"answerable": false` and a `fallback_reason`.

Interactive API docs: <http://localhost:8000/docs>. Other targets: `make test`, `make lint`,
`make typecheck`, `make eval`, `make logs`, `make down`.

> **The sample knowledge base** (`sample_docs/`) belongs to a fictional SaaS company,
> *Orbitask*: a 6-page PDF product guide (headings, a plan-limits table on page 5, running
> footers), a DOCX billing policy with heading levels and a pricing table, a Markdown FAQ, an
> HTML API troubleshooting guide with tables, and TXT release notes. The FAQ (last updated
> Nov 2025) says annual plans are refundable within **14 days**. The billing policy and
> release notes (March 2026) say **30 days**. That deliberate contradiction is discussed in
> [Limitations](#8-known-limitations-and-next-steps). Regenerate the PDF/DOCX with
> `make sample-docs`.

---

## 2. Retrieval approach and why

| Stage | What | Why |
|---|---|---|
| **Structure-aware chunking** | Chunks never cross a heading; about 500 tokens with about 80 tokens of sentence overlap; tables kept whole or split by rows with the header repeated; page numbers and the heading path (`Billing > Refunds`) stored per chunk | Precise citations; facts stay attached to the heading that gives them meaning |
| **Contextual chunk headers** | `title + "Section: …"` is prepended before embedding and full-text indexing; raw text is stored separately for display | A chunk that says "it is refundable within 30 days" still matches "annual plan refund" |
| **Query rewriting** | With session history, the LLM rewrites the latest message into a standalone query | "And on Business?" retrieves the Business rate limit |
| **Hybrid search** | Top 20 from pgvector cosine (HNSW) + top 20 from Postgres full-text (OR of stemmed lexemes, `ts_rank_cd`) | Dense search handles paraphrases; keyword search handles exact strings embeddings blur (`ORB-4091`, `X-Orbitask-Signature`) |
| **Reciprocal Rank Fusion** (k=60) | `score = Σ 1/(60 + rank)` | Combines two incomparable score scales without tuning |
| **Cross-encoder rerank** | `ms-marco-MiniLM-L-6-v2` reads query + passage together; keep top 5 | Much better top-1 precision than either retriever alone |
| **De-duplication** | Adjacent chunks of the same section are merged (overlap stitched out); duplicate texts dropped | No wasted context, no citing the same fact twice |
| **Confidence gate** | Skip the LLM if the best rerank score < `MIN_RERANK_SCORE` | Off-topic or injection input costs no LLM call |

**Ablation** (same 25 questions, same LLM and prompt; only the retriever changes):

| | vector only | hybrid (RRF) | hybrid + rerank (default) |
|---|---|---|---|
| Hit rate@1 | 85% | 80% | **90%** |
| Hit rate@5 | 100% | 100% | 100% |
| MRR | 0.925 | 0.900 | **0.950** |
| Answer accuracy (judge ≥ 4, all answerable Qs) | 95% | 100% | **100%** |
| Correctness, 1-5 | 4.85 | 4.95 | **5.00** |

On this small knowledge base (49 chunks), every configuration finds a relevant chunk in the top
5, so recall@5 saturates. The differences show up in *ranking* and *completeness*:
- Reranking gives the best top-1 precision and MRR.
- Vector-only missed the plan-limits table row on a multi-source question, so that answer
  was incomplete.
- Hybrid without reranking slightly hurt top-1 ordering on paraphrased questions; the
  reranker fixes that.

Expect the gaps to widen as the corpus grows. More near-duplicate passages compete for the
top 5, which is exactly where keyword matching and cross-encoder reranking help.

### Choosing the rerank threshold (`MIN_RERANK_SCORE = 0.0001`)

I calibrated the gate by running the eval with the gate disabled
([`eval/reports/calibration_gate_off.md`](eval/reports/calibration_gate_off.md)) and
reading the top rerank score of every question:

| Group | Top rerank score |
|---|---|
| Off-topic probes (weather, jokes, recipes) and the prompt-injection question | 1.3e-05 to 4.3e-05 |
| **Lowest-scoring answerable question** (paraphrase: "the API keeps telling me I'm sending too much traffic") | **3.7e-04** (next: 0.013) |
| Plausible-but-unanswerable questions (SLA, HIPAA, CEO, self-hosting) | 5.3e-04, 0.23, 0.26, **0.62** |

The cross-encoder is excellent at *ranking* but its absolute score cannot separate
"answerable" from "plausible but not covered". Catching the self-hosting question would need
a threshold ≥ 0.63, which would have refused about 40% of answerable questions. So the gate is
set **between the off-topic cluster and the lowest answerable score**: 2.3x above the
highest off-topic score and 3.7x below the lowest answerable one. Its job is to filter clearly off-topic input. Refusing plausible-but-uncovered
questions is the job of the next two layers, and the eval shows they do it: 5/5 correct
refusals and 0/20 false refusals. Re-calibrate whenever the corpus or reranker changes
([D4 in DECISIONS](docs/DECISIONS.md)).

---

## 3. Why pgvector

The knowledge base, the keyword index, chat sessions, query logs and feedback all live in
**one PostgreSQL 16 database**:

- **Hybrid search in one engine.** Vector similarity (HNSW) and full-text ranking (GIN
  `tsvector`) run over the same rows in SQL. There's no second store to keep in sync.
- **Transactional consistency.** Re-ingesting or deleting a document replaces or removes its
  chunks atomically, so a deleted policy can never be cited again.
- **Low operational cost at this scale.** A support knowledge base is thousands to low
  millions of chunks, well within pgvector's comfortable range on a modest instance. One
  database means one backup, one access-control model, one thing to monitor.
- **Logs next to data.** `query_logs` can be joined with `chunks` and `feedback` to debug a bad
  answer or find documentation gaps.

**When I'd switch to a dedicated vector DB (Qdrant, Pinecone, Milvus):** tens of millions
of vectors, very high query volume with strict p99 latency targets, heavy metadata
filtering across many tenants, or a need for managed horizontal scaling. The `Retriever`
only depends on two search functions (`vector.py`, `keyword.py`), so swapping the vector
side is a contained change.

---

## 4. How hallucinations are handled

Each layer fails *closed*: when in doubt, the system returns the fallback, never a guess.

| # | Layer | What it does | Code |
|---|---|---|---|
| 1 | Retrieval quality | Hybrid + RRF + rerank puts the right passage in front of the model | [`app/retrieval/retriever.py`](app/retrieval/retriever.py) |
| 2 | Confidence gate | Best rerank score below threshold → fallback without calling the LLM | [`app/generation/guardrails.py`](app/generation/guardrails.py) `retrieval_gate` |
| 3 | Constrained prompt | Answer only from `<source>` blocks; cite every claim `[S#]`; set `answerable=false` if not covered; source text is data, not instructions; flag conflicting sources | [`app/generation/prompts.py`](app/generation/prompts.py) |
| 4 | History isolation | Chat history is used only to rewrite the query, never shown to the answering prompt | [`app/retrieval/query_rewriter.py`](app/retrieval/query_rewriter.py) |
| 5 | Structured output | OpenAI strict JSON schema / Anthropic forced tool call → `{answer, citations, answerable, confidence}`, validated by Pydantic; one retry on invalid output | [`app/generation/answerer.py`](app/generation/answerer.py) |
| 6 | Citation validation | Citation ids must exist in the provided context; invented `[S7]` markers are stripped; an "answerable" answer with no valid citation is rejected | [`app/generation/citations.py`](app/generation/citations.py) |
| 7 | Single fallback | Every failure returns the same HTTP 200 shape with `fallback_reason` ∈ `low_retrieval_confidence`, `model_unanswerable`, `invalid_citations`, `llm_error`. LLM outages never become a 500. | [`app/services/chat.py`](app/services/chat.py) |
| 8 | Injection hardening | `<` in documents is escaped so a document can't close its own `<source>` tag | [`app/generation/answerer.py`](app/generation/answerer.py) `render_sources` |

Every turn is persisted to `query_logs`:
- the original and rewritten query;
- every candidate's vector, keyword and rerank scores, and the final chunk ids;
- which sources were cited;
- the fallback reason, model and prompt version;
- token usage and per-stage latency (rewrite, retrieve, rerank, generate).

So "why did it say that?" is answerable with one SQL query. `POST /v1/chat/{message_id}/feedback`
records thumbs up or down.

---

## 5. Evaluation

`make eval` runs [`eval/dataset.yaml`](eval/dataset.yaml) (25 questions) through the **same
`ChatService` the API uses**, once per retrieval configuration. It writes a Markdown and a
JSON report to `eval/reports/`. A committed run: [`eval/reports/sample_report.md`](eval/reports/sample_report.md).

| Category | n | Result (hybrid + rerank) |
|---|---|---|
| Single-fact lookup | 8 | 8/8 correct |
| Multi-chunk / multi-document synthesis | 4 | 4/4 correct |
| Table / later-page facts | 3 | 3/3 correct |
| Paraphrased (little keyword overlap) | 3 | 3/3 correct |
| Multi-turn follow-ups | 2 | 2/2 correct |
| Unanswerable (plausible, not in docs) | 4 | 4/4 refused (`model_unanswerable`) |
| Prompt injection / off-topic | 1 | 1/1 refused (`low_retrieval_confidence`, no LLM call) |

| Metric (hybrid + rerank) | Value |
|---|---|
| Retrieval hit rate@5 / @1, MRR | 100% / 90%, 0.950 |
| Citation accuracy (answers citing an expected source) | 100% |
| Correctness (LLM judge, 1-5) | 5.00 |
| Faithfulness (LLM judge, 1-5) | 4.90 |
| Correct refusals on unanswerable | 100% (5/5) |
| False refusals on answerable | 0% (0/20) |
| Latency p50 / p95 (end to end, CPU models, gpt-4.1-mini) | 1.61 s / 2.64 s |
| Avg tokens per query | 1,085 |

Generator: `gpt-4.1-mini`. Judge: `gpt-4.1`, a different and stronger model to reduce
self-preference bias. The judge scores correctness against the reference answer and
faithfulness against the retrieved passages, and gives a one-sentence reason.

**Read these numbers honestly.** 25 questions over 49 chunks written by the same person who
wrote the questions is a *smoke test of the pipeline's behaviour*, not a benchmark. Perfect
scores here mean the mechanisms work (refusal, citation, rewriting, table extraction), not
that real-world accuracy is 100%. The first thing to do with a client's real documents is
build a 100-300 question set from real support tickets.

**Adding questions.** Append to `eval/dataset.yaml`:

```yaml
- id: sf-09
  category: single_fact          # see header of dataset.yaml for categories
  question: How long are export download links valid?
  expected_answer: 7 days.
  expected_sources:
    - {file: orbitask_faq.md, section: export}   # section = substring of the heading path; page optional
  answerable: true
  # history: ["earlier user turn", ...]          # for multi-turn questions
```

Useful flags: `--only sf-09,mt-01`, `--no-judge`, `--min-rerank-score 0` (calibration),
`--judge-model gpt-4.1`, `--render eval/reports/<file>.json` (re-render without re-running).

---

## 6. API reference and integration

All `/v1` routes need the `X-API-Key` header. Requests can send `X-Request-ID`; it is echoed
back and attached to every log line and to the `query_logs` row.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat` | Ask a question. Body `{ "message": str, "session_id"?: uuid }`. Omit `session_id` to start a conversation. |
| `POST` | `/v1/chat/{message_id}/feedback` | `{ "helpful": bool, "comment"?: str }` |
| `GET` | `/v1/sessions/{id}/messages` | Conversation history, including the sources and fallback reason of each answer |
| `DELETE` | `/v1/sessions/{id}` | Delete a conversation (and its logs) |
| `POST` | `/v1/documents` | Multipart upload (`file`). `202` new, `200` identical content already ingested, `415` unsupported type, `413` too large |
| `GET` | `/v1/documents`, `/v1/documents/{id}` | List documents / poll status: `processing`, `ready` or `failed` (with `error`) |
| `DELETE` | `/v1/documents/{id}` | Remove a document and all its chunks |
| `GET` | `/health`, `/ready` | Liveness / readiness (DB reachable, pgvector installed, models loaded) |

**Python**

```python
import httpx

client = httpx.Client(
    base_url="https://support-ai.example.com", headers={"X-API-Key": "..."}, timeout=30
)


def ask(message: str, session_id: str | None = None) -> dict:
    r = client.post("/v1/chat", json={"message": message, "session_id": session_id})
    r.raise_for_status()
    return r.json()


reply = ask("Can I get a refund on an annual plan?")
if reply["answerable"]:
    print(reply["answer"])
    for s in reply["sources"]:
        print(f"  [{s['id']}] {s['filename']} > {s['section']} (page {s['page']})")
else:
    # fallback_reason: low_retrieval_confidence | model_unanswerable | invalid_citations | llm_error
    open_support_ticket(question=..., reason=reply["fallback_reason"])
follow_up = ask("What about monthly plans?", session_id=reply["session_id"])
```

**JavaScript / TypeScript**

```js
async function ask(message, sessionId) {
  const res = await fetch("https://support-ai.example.com/v1/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-API-Key": process.env.GROUNDEDRAG_KEY },
    body: JSON.stringify({ message, session_id: sessionId ?? null }),
  });
  if (!res.ok) throw new Error(`GroundedRAG ${res.status}`);
  return res.json(); // { answer, answerable, confidence, fallback_reason, sources[], session_id, message_id }
}

const reply = await ask("How do I reset my password?");
showAnswer(reply.answer, reply.sources);               // render [S1] markers as links to sources
if (!reply.answerable) offerHumanHandoff(reply.fallback_reason);
```

---

## 7. Project layout

```
app/
  main.py            FastAPI app factory; lifespan loads models once
  config.py          pydantic-settings (every env var below)
  components.py      long-lived singletons (engine, embedder, reranker, LLM client)
  api/               routers (chat, documents, sessions, health), schemas, DI dependencies
  ingestion/         parsers/ (pdf, docx, text+md+html), chunker, cleaning, pipeline
  retrieval/         embedder, vector, keyword, hybrid (RRF), reranker, dedupe, query_rewriter, retriever
  generation/        prompts (versioned), answerer, citations, guardrails
  llm/               provider-agnostic interface + OpenAI and Anthropic clients
  services/chat.py   one chat turn end to end + persistence
  observability/     JSON logging, request-id middleware
  testing.py         offline fakes (embedder, reranker, LLM) for tests / keyless demos
migrations/          Alembic (pgvector extension, HNSW + GIN indexes)
eval/                dataset.yaml, run_eval.py, metrics.py, reports/
sample_docs/         the Orbitask knowledge base
scripts/             make_sample_docs.py, ingest_samples.py
tests/               unit/ (offline) + integration/ (testcontainers pgvector)
docs/DECISIONS.md    design decisions (ADR-style)
```

Quality gates: `make lint` (ruff), `make typecheck` (mypy `--strict` on `app/`), `make test`
(80 tests, 97% coverage on `app/`; integration tests start `pgvector/pgvector:pg16` with
testcontainers and use fake models, so they need Docker but no API key).

---

## 8. Known limitations and next steps

- **Contradicting documents.** The FAQ says 14 days, the newer policy says 30. Today the prompt
  tells the model to flag conflicts and prefer the source with the later *stated* effective
  date. In the eval it answered "30 days" and cited the policy and the release notes. But this
  relies on dates appearing in the text. Next step: per-document `effective_date` / `supersedes`
  metadata, recency-aware ranking, and a report of conflicting chunks for content owners.
- **Scanned PDFs / images.** Documents without a text layer fail with a clear "needs OCR" error.
  Next step: OCR (Tesseract or a vision model) and figure captioning.
- **Small, self-authored eval set.** See [Evaluation](#5-evaluation). Build a larger set from
  real tickets and track metrics per release in CI.
- **Absolute relevance scores.** The cross-encoder can't tell "plausible but uncovered" from
  "answerable" (see threshold calibration), so refusal of those relies on the LLM. A
  fine-tuned answerability classifier or a stronger reranker (e.g. `bge-reranker-v2`) would
  add an independent check.
- **Multilingual.** The embedding model, reranker and the `english` text-search config are
  English-only. Next step: `bge-m3` / multilingual reranker, a per-language FTS config, and
  answering in the user's language.
- **Per-tenant isolation.** A single shared knowledge base. Next step: a `tenant_id` on documents
  and chunks, enforced by Postgres row-level security, plus per-tenant API keys.
- **Streaming.** Responses are returned whole. Server-sent events would improve perceived
  latency. Citation validation would then run on the final buffer before the answer is marked
  final.
- **Async ingestion queue.** FastAPI BackgroundTasks run in-process. A restart mid-ingestion
  leaves a document in `processing`. Next step: a durable queue (Arq/Celery/SQS) with retries and
  a stale-job sweeper.
- **Business tools.** Phase 2 is to give the assistant tools behind the same guardrails:
  ticketing (Zendesk/Freshdesk: create a ticket on fallback), CRM lookups (plan, seats, billing
  status, so "my plan" is resolved), and order and subscription status, with the same logging
  and evaluation discipline.
- **Auth.** A static API-key list is fine for service-to-service use. Production would add key
  rotation, per-key rate limits and audit logs.

---

## 9. Configuration reference

All settings are environment variables (also read from `.env`). See [`.env.example`](.env.example).

| Variable | Default | What it does |
|---|---|---|
| `API_KEYS` | *(empty)* | Comma-separated keys accepted in `X-API-Key`. Empty = every `/v1` request is rejected. |
| `LOG_LEVEL` | `INFO` | Root log level (JSON logs to stdout) |
| `MAX_UPLOAD_MB` | `20` | Upload size limit (HTTP 413 above it) |
| `DATABASE_URL` | `postgresql+asyncpg://groundedrag:groundedrag@localhost:5433/groundedrag` | Async SQLAlchemy URL. docker-compose overrides it for the API container. |
| `DB_POOL_SIZE` | `5` | Connection pool size |
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, `ollama`, or `fake` (offline stub). `openai`/`anthropic` without their API key fall back to `ollama`. |
| `LLM_MODEL` | *(provider default)* | `gpt-4.1-mini` for OpenAI, `claude-sonnet-5-5` for Anthropic. Ignored on the Ollama fallback. |
| `LLM_TEMPERATURE` | `0` | Sampling temperature. Empty = don't send (for models that only accept the default). Not sent to Anthropic. |
| `LLM_MAX_TOKENS` | `800` | Max output tokens for the answer |
| `LLM_TIMEOUT_S` | `30` | Per-request LLM timeout; on expiry the turn falls back with `llm_error` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | *(empty)* | Provider credentials |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama's OpenAI-compatible endpoint. docker-compose sets `http://host.docker.internal:11434/v1` for the API container (override with `OLLAMA_DOCKER_BASE_URL`). |
| `OLLAMA_MODEL` | `Qwen3-Coder:latest` | Ollama model used by `LLM_PROVIDER=ollama` and by the fallback |
| `OLLAMA_TIMEOUT_S` | `120` | Per-request timeout for Ollama (local models are slower; the first call loads the model) |
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers), `openai`, or `fake` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model (baked into the Docker image) |
| `EMBEDDING_DIM` | `384` | Vector size. Must match the model **and** the pgvector column (fixed at first migration). |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Used when `EMBEDDING_PROVIDER=openai` (truncated to `EMBEDDING_DIM`) |
| `CHUNK_MAX_TOKENS` | `500` | Target maximum chunk size (approximate tokens) |
| `CHUNK_OVERLAP_TOKENS` | `80` | Sentence overlap between consecutive chunks of a section |
| `RETRIEVAL_MODE` | `hybrid_rerank` | `vector`, `hybrid` or `hybrid_rerank` (the eval ablation uses all three) |
| `VECTOR_TOP_K` / `KEYWORD_TOP_K` | `20` / `20` | Candidates from each retriever before fusion |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `FINAL_TOP_K` | `5` | Sources given to the LLM after rerank and de-dupe |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used for reranking |
| `MIN_RERANK_SCORE` | `0.0001` | Confidence gate (see [threshold calibration](#choosing-the-rerank-threshold-min_rerank_score--00001)) |
| `HISTORY_TURNS` | `3` | Previous exchanges used for query rewriting (`0` disables rewriting) |
| `FALLBACK_MESSAGE` | *"I'm sorry, I couldn't find a reliable answer…"* | Text returned whenever the system falls back |
| `API_HOST_PORT` / `POSTGRES_HOST_PORT` | `8000` / `5433` | Host ports in docker-compose |
