# GeekBrain RAG Operations

## Production contract

The system follows one controlled path: **source → clean → chunk → index → publish**.

1. Source files live in `data_package/knowledge_base`. Each source receives an owner, status,
   version, checksum, review date and expiry in the generated registry.
2. `scripts/ingest.py` normalizes UTF-8, validates status/size/freshness, scans for document-level
   prompt-injection indicators and creates Bedrock metadata sidecars.
3. Bedrock uses fixed 500-token chunks with 15% overlap. Hierarchical chunking is intentionally
   avoided because it is a poor fit for S3 Vectors metadata limits.
4. Validated `CURRENT` and explicitly governed `DRAFT` files are uploaded to the active prefix. The
   runtime filters to `CURRENT` unless the question explicitly asks about a plan/proposal/draft. S3
   versioning retains prior object versions for rollback. Archived files stay outside production.
5. A Bedrock ingestion job indexes the latest publish into a dedicated S3 vector index. The
   manifest is written outside the ingestion prefix for audit and reproducibility.

Run:

```powershell
.\venv\Scripts\python.exe scripts\provision_aws.py
.\venv\Scripts\python.exe scripts\ingest.py
.\venv\Scripts\python.exe scripts\evaluate.py --mode retrieval --levels 1,2
.\venv\Scripts\python.exe scripts\evaluate.py --mode answer --levels 3 --output build\answer-evaluation-l3-final.json
.\venv\Scripts\python.exe scripts\evaluate.py --mode answer --levels 4 --output build\answer-evaluation-l4-final.json
.\venv\Scripts\python.exe scripts\evaluate.py --mode answer --levels 5 --output build\answer-evaluation-l5-final.json
```

Level 4 keeps one checkpoint thread per conversation. Level 5 uses the stricter 0.9 judge threshold.
Use `--case-id L4-04` (or another exact ID) for a focused diagnostic rerun.

## Ownership and freshness

Rules are in `config/governance.yaml`. The migration review baseline is explicit and does not use
checkout timestamps. Living documents expire after their configured cadence; archived documents
are excluded from the production prefix. The daily CI job checks expiry. At 30 days before expiry,
the owner should review the source and set an explicit `reviewed_at` value in frontmatter. Expired
sources block normal publishing and are filtered at retrieval time.

Recommended cadence:

- API, security, incident, deployment, SLA and runbook sources: every 180 days or immediately after
  a breaking change/incident.
- Service/team/capacity sources: every 180 days and after ownership or architecture changes.
- Postmortems and historical reviews: immutable; create a new version/correction rather than
  silently rewriting facts.
- Retrieval and answer evaluation: on every change. Adversarial and live-tool suites: weekly.

## Accuracy and failure policy

- Retrieval is semantic because the store is S3 Vectors; `HYBRID` is not assumed.
- Every factual statement must cite numbered evidence. Invalid citations trigger one repair pass,
  then abstention.
- Current documents outrank archived ones. Conflicts must be surfaced with version/date, not hidden.
- Expired documents are not silently used. Missing KB, unavailable monitoring, empty DB results and
  guardrail interventions produce a scoped abstention.
- Live metrics are labeled live; SQLite data is labeled historical. The synthesizer must not merge
  their timestamps as if they were equivalent.
- Cross-source SLA, historical-average, capacity and deadline comparisons are computed
  deterministically and published as citation-ready `DERIVED` evidence. Holistic investigations
  use a deterministic extractive digest and bounded read-only analytics subqueries.
- User text and retrieved documents are always treated as untrusted data, never instructions.

## Agent safety and edge cases

- Multi-intent requests can route to documents, analytics DB and live monitoring in the same turn.
- Tool access is allowlisted by detected intent. There is no generic network or shell tool.
- Database generation accepts one parameterized `SELECT`; mutations, PRAGMA, UNION, comments,
  unknown tables, long execution and more than 100 rows are denied by validation plus SQLite's
  authorizer and query-only mode.
- Monitoring service names come from a fixed allowlist and URLs are encoded.
- Bedrock Guardrails blocks prompt attacks and credentials and checks answer grounding/relevance.
- Conversation rewriting preserves all intents and resolves pronouns, but never answers or follows
  instructions embedded in history.
- Requests with two clauses, conflicts, stale sources, empty retrieval, invalid model output,
  duplicate chunks, archived/current versions, tool timeout and citation spoofing are covered by
  deterministic behavior and tests.

## Rollback and incident response

1. Stop publication; do not delete the vector bucket.
2. Restore the last good S3 object versions under `published/current/`.
3. Run `scripts/ingest.py --no-sync` to validate, then trigger a normal ingestion.
4. Run retrieval plus answer evaluation before reopening traffic.
5. Inspect `rag_ops.db` for query audit metadata. Prompts and answers are deliberately not persisted;
   only hashes, intents, sources, latency, citation counts and abstention are stored.

Never remove the old vector index until the new KB passes evaluation and rollback is verified.
