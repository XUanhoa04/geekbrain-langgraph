# Kiến trúc và luồng xử lý

## System architecture — Graphviz

SVG dưới đây được render từ
[`geekbrain-rag-architecture.dot`](diagrams/geekbrain-rag-architecture.dot) bằng Graphviz.

![GeekBrain RAG architecture](diagrams/geekbrain-rag-architecture.svg)

## End-to-end request flow — Mermaid

```mermaid
%%{init: {"theme": "base", "themeVariables": {
  "fontFamily": "Inter, Segoe UI, sans-serif",
  "primaryTextColor": "#10233f",
  "lineColor": "#526d82"
}}}%%
flowchart LR
    U([User question]):::user --> IG{Input guardrail}:::guard
    IG -->|blocked| STOP([Safe refusal]):::blocked
    IG -->|allowed| R{Multi-intent router}:::router

    R -->|DOCUMENT| KB[Bedrock KB<br/>S3 Vectors]:::document
    R -->|DATABASE| DB[Safe analytics<br/>SELECT only]:::database
    R -->|LIVE_METRICS| LIVE[Monitoring API<br/>bounded allowlist]:::live

    KB --> G[Evidence gatherer]:::core
    DB --> G
    LIVE --> G

    G --> F{Fresh + valid?}:::decision
    F -->|no evidence| AB([Scoped abstention]):::blocked
    F -->|yes| D[Deterministic DERIVED evidence<br/>SLA · average · capacity · deadline]:::derived
    D --> S[Grounded synthesis<br/>Nova Pro + Lite fallback]:::model
    S --> C{Valid citations?}:::decision
    C -->|repair fails| AB
    C -->|valid| OG{Contextual grounding<br/>cited evidence only}:::guard
    OG -->|blocked| RETRY[Short fact-only retry]:::model
    RETRY --> OG
    OG -->|pass| A([Cited answer]):::success
    A --> AUDIT[(Privacy-preserving audit<br/>hashes + metadata only)]:::audit

    classDef user fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e,stroke-width:2px;
    classDef guard fill:#fff1f2,stroke:#e11d48,color:#881337,stroke-width:2px;
    classDef blocked fill:#ffe4e6,stroke:#be123c,color:#881337,stroke-width:2px;
    classDef router fill:#f3e8ff,stroke:#9333ea,color:#581c87,stroke-width:2px;
    classDef document fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:2px;
    classDef database fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:2px;
    classDef live fill:#cffafe,stroke:#0891b2,color:#164e63,stroke-width:2px;
    classDef core fill:#e0e7ff,stroke:#4f46e5,color:#312e81,stroke-width:2px;
    classDef decision fill:#f8fafc,stroke:#64748b,color:#1e293b,stroke-width:2px;
    classDef derived fill:#fae8ff,stroke:#c026d3,color:#701a75,stroke-width:2px;
    classDef model fill:#ede9fe,stroke:#7c3aed,color:#4c1d95,stroke-width:2px;
    classDef success fill:#d1fae5,stroke:#059669,color:#064e3b,stroke-width:3px;
    classDef audit fill:#e2e8f0,stroke:#475569,color:#0f172a,stroke-width:2px;
```

## Cross-source grounding sequence — Mermaid

```mermaid
%%{init: {"theme": "base", "themeVariables": {
  "actorBkg": "#ede9fe", "actorBorder": "#7c3aed",
  "actorTextColor": "#3b0764", "signalColor": "#334155",
  "activationBkgColor": "#cffafe", "activationBorderColor": "#0891b2"
}}}%%
sequenceDiagram
    autonumber
    actor User
    participant Agent as LangGraph Agent
    participant KB as Bedrock KB / S3 Vectors
    participant DB as Safe Analytics DB
    participant Mon as Monitoring API
    participant Derive as Deterministic Verifier
    participant GR as Bedrock Guardrail

    User->>Agent: Compare current p99 with Q1 average
    par Governed document context when needed
        Agent->>KB: Semantic retrieve + CURRENT filter
        KB-->>Agent: Cited chunks + governance metadata
    and Historical baseline
        Agent->>DB: Allowlisted parameterized SELECT
        DB-->>Agent: Q1 average JSON
    and Live observation
        Agent->>Mon: GET /metrics/{allowlisted-service}
        Mon-->>Agent: Current metric JSON + timestamp
    end
    Agent->>Derive: Join same service + metric
    Derive-->>Agent: current, baseline, difference, %, direction
    Agent->>GR: Answer + only cited evidence
    alt grounded and relevant
        GR-->>Agent: PASS
        Agent-->>User: Concise answer with citations
    else unsupported claim
        GR-->>Agent: BLOCK
        Agent-->>User: Fact-only retry or abstention
    end
```

## Governed publishing flow — Mermaid

```mermaid
flowchart TD
    SRC[Source documents]:::source --> N[Normalize UTF-8]:::step
    N --> META[Build owner/version/checksum/<br/>review/expiry/status metadata]:::step
    META --> SCAN{Prompt-injection<br/>or stale?}:::check
    SCAN -->|suspicious| Q[Quarantine]:::bad
    SCAN -->|expired CURRENT| HOLD[Block publication]:::bad
    SCAN -->|valid| STATUS{Status}:::check
    STATUS -->|ARCHIVED| ARCH[Exclude from production]:::archive
    STATUS -->|CURRENT or governed DRAFT| S3[(Versioned source bucket<br/>published/current/)]:::store
    S3 --> ING[Bedrock ingestion job]:::aws
    ING --> V[(S3 Vector bucket<br/>1024-d Titan embeddings)]:::vector
    V --> TEST[Retrieval + answer evaluation]:::test
    TEST -->|pass| PROD([Production KB]):::good
    TEST -->|fail| ROLLBACK[Restore prior S3 versions]:::bad

    classDef source fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e;
    classDef step fill:#ede9fe,stroke:#7c3aed,color:#4c1d95;
    classDef check fill:#fef3c7,stroke:#d97706,color:#78350f;
    classDef bad fill:#ffe4e6,stroke:#e11d48,color:#881337;
    classDef archive fill:#e2e8f0,stroke:#64748b,color:#334155;
    classDef store fill:#dcfce7,stroke:#16a34a,color:#14532d;
    classDef aws fill:#ffedd5,stroke:#ea580c,color:#7c2d12;
    classDef vector fill:#cffafe,stroke:#0891b2,color:#164e63;
    classDef test fill:#fae8ff,stroke:#c026d3,color:#701a75;
    classDef good fill:#d1fae5,stroke:#059669,color:#064e3b,stroke-width:3px;
```
