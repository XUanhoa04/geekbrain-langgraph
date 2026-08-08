# Đánh giá GeekBrain RAG Agent

## Tóm tắt

Đây không còn là một demo “retrieve rồi nhét context vào prompt”. Điểm đáng chú ý nhất là agent
đã biến các ràng buộc thường chỉ nằm trong prompt thành cơ chế có thể kiểm tra: governance trước
ingestion, SQL sandbox, multi-source evidence, arithmetic deterministic, citation validation,
contextual grounding và privacy-preserving audit.

Agent phù hợp nhất với trợ lý vận hành nội bộ cần trả lời đồng thời từ tài liệu, dữ liệu lịch sử và
telemetry hiện tại—ví dụ đánh giá SLA, điều tra incident, so sánh xu hướng và lập reliability report.

## Những bài toán RAG phổ biến đã giải quyết

| Bài toán | Cách xử lý | Mức độ |
|---|---|---|
| Chỉ tìm được tài liệu, không trả lời được số liệu live/DB | Router kích hoạt đồng thời document, database và monitoring | Đã giải quyết trong phạm vi 3 nguồn |
| Model tự tính sai khi ghép hai nguồn | Tạo `DERIVED` evidence cho current/baseline/difference/%/direction/verdict | Đã giải quyết cho các phép tính đã định nghĩa |
| Citation có nhưng không chứng minh claim | Kiểm tra citation in-range, chỉ gửi cited evidence vào grounding guardrail | Giảm mạnh, chưa phải formal proof |
| Prompt injection trong tài liệu | Scan và quarantine trước publish; evidence luôn là untrusted data | Đã có hai lớp phòng vệ |
| Arbitrary SQL và SQL injection | SELECT-only, allowlist table/column/function, bind parameters, authorizer, query-only, step/row/complexity limit | Đã giải quyết tốt cho SQLite analytics |
| Tài liệu stale/draft/archive lẫn nhau | Status allow-list, parsed expiry, malformed-metadata rejection, explicit draft intent | Đã giải quyết theo policy hiện tại |
| Câu hỏi nhiều ý bị route một nguồn | Multi-intent deterministic routing và parallel gather | Đã giải quyết cho intent đã biết |
| Hội thoại mất chủ thể | Memory checkpoint, standalone rewrite và deterministic service anchoring | Đã giải quyết phần lớn benchmark |
| Lỗi nguồn làm model đoán | Provider error evidence, bounded timeout và scoped abstention | Đã giải quyết theo fail-closed |
| Audit làm lộ dữ liệu | Chỉ lưu query hash, intent, source, latency, citation, abstention | Đã giải quyết ở audit DB của agent |
| Guardrail chặn phép suy luận đúng | Citation-ready deterministic digest/answer-ready facts | Đã giải quyết cho SLA, average, capacity, deadline và holistic benchmark |
| Live metric có jitter | Evaluator chấp nhận tolerance và hướng so sánh theo observation thật | Đã giải quyết ở evaluation |

## Điểm đặc biệt

1. **S3 Vectors-native:** kiến trúc Bedrock KB dùng S3 Vectors, không phụ thuộc OpenSearch.
2. **Evidence là data product:** evidence có kind, source, score, metadata, citation ID và lineage;
   phép tính cũng trở thành evidence thay vì chain-of-thought ẩn.
3. **Deterministic where it matters:** router, SQL plans quan trọng, arithmetic, capacity proximity,
   overdue verdict và holistic summary đều có đường deterministic.
4. **Guardrail-friendly by construction:** agent không hạ threshold để làm test xanh; nó cấu trúc lại
   evidence và rút gọn claim để guardrail kiểm chứng được.
5. **Governance xuyên suốt lifecycle:** cùng một status/freshness contract được dùng khi clean,
   publish, retrieve và rollback.
6. **Evaluation nhiều tầng:** retrieval, answer, grounded computation, conversation, holistic và
   resilience được đo riêng thay vì chỉ dùng một accuracy score.

## Những vấn đề chưa giải quyết hoàn toàn

### 1. Không có bảo đảm hình thức rằng mọi claim đều được citation chứng minh

Regex xác nhận citation tồn tại và Bedrock contextual grounding cho điểm xác suất. Đây chưa phải
claim-level entailment verifier. Một câu dài có nhiều mệnh đề vẫn có thể được một citation hỗ trợ
chỉ một phần.

**Nên làm tiếp:** tách answer thành structured claims, map từng claim tới evidence IDs, chạy NLI hoặc
automated reasoning per claim và từ chối riêng claim không đạt.

### 2. Router vẫn là taxonomy hữu hạn

Router hiện normalize dấu tiếng Việt, có phrase set EN/VI, tolerance cho typo phổ biến và nhận diện
service mới theo naming convention. Analytics còn đọc catalog service thực từ database. Tuy vậy đây
vẫn là classifier deterministic với taxonomy hữu hạn; cách diễn đạt hoặc intent hoàn toàn mới có thể
bị bỏ sót. Việc không dùng LLM làm router mặc định là chủ ý để giữ latency, cost và hành vi fail-closed,
không phải bảo đảm coverage tuyệt đối.

**Nên làm tiếp:** schema-based intent classifier có confidence, out-of-distribution detection và
golden routing set mở rộng bằng paraphrase/adversarial generation.

### 3. Deterministic derivation chưa phải engine tính toán tổng quát

Các phép SLA, average, growth, capacity, deadline và một số holistic plans đã an toàn, nhưng câu hỏi
phân tích mới vẫn có thể cần phép tính chưa được định nghĩa. Không nên quay lại arbitrary code/SQL.

**Nên làm tiếp:** DSL tính toán allowlisted, typed units, provenance graph và independent arithmetic
verification.

### 4. SQLite phù hợp demo/single-node, chưa phải production analytics backend

Authorizer và query-only bảo vệ tốt nhưng SQLite không giải quyết concurrency, HA, row-level security,
network isolation hay warehouse-scale workloads.

**Nên làm tiếp:** chuyển interface hiện tại sang read replica/Athena/Redshift với IAM, query budget,
workgroup controls và statement timeout; vẫn giữ semantic plan allowlist.

### 5. Monitoring API hiện tuần tự theo service

So sánh toàn bộ service gọi nhiều endpoint tuần tự trong một session. Timeout được bounded nhưng
latency tăng tuyến tính và chưa có circuit breaker/cache/rate-limit strategy.

**Nên làm tiếp:** async bounded concurrency, per-provider circuit breaker, short TTL cache và partial
result health metadata.

### 6. Memory là in-process

`MemorySaver` phù hợp test nhưng không bền qua restart, không chia sẻ giữa replicas và chưa có chính
sách retention/tenant isolation.

**Nên làm tiếp:** durable encrypted checkpointer, tenant-scoped keys, TTL, deletion API và memory
redaction.

### 7. Chưa có end-to-end observability đầy đủ

Audit đã privacy-safe nhưng chưa có distributed tracing cho mỗi provider, token/cost budget, guardrail
score trends, retrieval drift hoặc SLO dashboard của chính agent.

**Nên làm tiếp:** OpenTelemetry traces không chứa raw content, CloudWatch EMF metrics, cost/latency
budgets và alarms cho abstention/grounding degradation.

### 8. Chưa kiểm chứng đa ngôn ngữ end-to-end và tải lớn

Router và credential policy đã có targeted Vietnamese/adversarial unit tests, nhưng answer benchmark
vẫn chủ yếu tiếng Anh. Chưa có Vietnamese retrieval/grounding golden set, load, soak, concurrency,
chaos hoặc regional failover test.

**Nên làm tiếp:** Vietnamese golden set, Locust/k6 load test, dependency fault injection và Bedrock
quota/throttling tests.

### 9. Guardrail và model vẫn là dependency xác suất

Các renderer deterministic giảm variance nhưng synthesis mở vẫn phụ thuộc model và guardrail score.
Model fallback đã cấu hình, chưa có continuous canary hoặc automatic rollback theo quality metrics.

**Nên làm tiếp:** shadow evaluation, canary model rollout, score distribution monitoring và quality
gate trước đổi model/prompt/guardrail version.

### 10. Public-repo hygiene không thay thế secret scanning trong CI

`.gitignore` ngăn file local phổ biến nhưng không ngăn developer paste secret vào source rồi commit.

**Nên làm tiếp:** GitHub secret scanning/push protection, Gitleaks, dependency review, CodeQL và
pre-commit hooks.

## Kết luận

Agent đã giải quyết tốt nhóm vấn đề khó nhất của enterprise RAG trong một domain đóng: governance,
multi-source retrieval, safe structured analytics, live-vs-historical comparison, citations,
freshness, prompt injection và fail-closed behavior. Nó chưa phải một nền tảng RAG tổng quát có
formal correctness, durable multi-tenant memory và production-scale observability. Kiến trúc hiện
tại là nền tảng mạnh để tiến tới các mục tiêu đó mà không phải phá bỏ safety boundaries đã có.
