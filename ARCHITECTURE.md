# Architecture of AlreadyMentioned

Compact map for agents and developers. See [docs/DEVELOPMENT.ru.md](docs/DEVELOPMENT.ru.md) for setup and file details, and [docs/GUIDE.ru.md](docs/GUIDE.ru.md) for user-visible behavior.

## Runtime and boundaries

```text
Telegram long polling
  → main.py: Dispatcher and dependency wiring
  → handlers/: commands, text messages, callback buttons
  → services/: question filter, temporary caches, embeddings, search
  → repositories/: chat, solution, feedback queries
  → database/: SQLite connection and schema
```

`src/already_mentioned/main.py` is the entry point (`python -m already_mentioned.main` or installed `already-mentioned`). It registers routers in this order: `start`, `solve`, `management`, `messages`, `feedback`. It creates one SQLite connection and supplies repositories, search, embedding service, and in-memory caches to aiogram handlers. `config.py` reads `.env` and environment variables. `bot/factory.py` creates the Telegram client, including an HTTPS proxy if configured; `bot/permissions.py` checks chat administrator status.

| Layer | Main files | Role |
| --- | --- | --- |
| Telegram | `handlers/start.py`, `solve.py`, `management.py`, `messages.py`, `feedback.py` | Validate Telegram context and permissions; invoke services and repositories; send messages or callbacks. |
| Search | `services/questions.py`, `embeddings.py`, `similarity.py`, `search.py` | Select likely Russian questions, encode them, compare with saved questions and answers from the same chat, apply the chat threshold. |
| Temporary state | `services/reply_cache.py`, `confirmations.py` | Hold reply chains for optional `/solve` and `/forget` confirmations in memory; state is lost on restart. |
| Persistence | `repositories/chats.py`, `solutions.py`, `feedback.py`, `database/connection.py`, `schema.py`, `backup.py`, `models/entities.py` | SQLite access, versioned schema, online backup CLI, and returned dataclasses. `models/entities.py` is not the ML model. |
| Error logs | `error_logging.py`, `handlers/solve.py`, `messages.py` | Record operation, chat/message IDs and error type without message or vector contents. |
| Diagnostics and experiments | `diagnostics.py`, `evaluation.py`, `evaluation_cases.py`, `historical_evaluation.py`, `historical_model_comparison.py`, `reranking_evaluation.py`, `services/reranking.py`, `model_comparison.py`, `offline_training.py`, `public_model_comparison.py`, `candidate_model_comparison.py` | Read-only CLI checks, fixed-case and public-data model comparison, historical scoring/reranking experiments, and public-data adapter training. Experiment outputs are isolated from bot search. |

## Main flows

**Save a solution:** A group member posts a text question → an administrator replies with a text answer → `messages.py` calls `solve.remember_admin_answer()` and stores the candidate in `ReplyCache` → for a question passing `questions.py` and with two direct links, `autosave_admin_answer()` encodes the original question and answer separately and persists the pair silently. The administrator can optionally reply to their own answer with `/solve` within 6 hours to save a question rejected by the filter. Direct message links make supergroups the practical target.

**Suggest an answer:** `messages.py` ignores private chats, bots, commands and non-text messages → `questions.py` filters likely questions → `SearchService` loads saved solutions of the current chat only → `EmbeddingService.embed_query()` encodes the new question → `similarity.py` compares it with each saved question and answer vector → the highest score is returned only when it meets the chat threshold → the handler replies to the new question with the saved answer as an escaped HTML blockquote, feedback buttons, and a source-link button. The preview is capped at 3000 characters. This is a linear scan of the chat's solutions. Feedback is stored, not used for ranking.

**Manage chat data:** `/status` reads count and threshold; `/threshold` changes that chat's threshold after an admin check. `/solutions` shows a paginated, chat-scoped administrator list with vote totals. `/editquestion ID` and `/editanswer ID` require an admin reply to the stored answer and matching solution ID; each edit recalculates the `passage:` embedding of the changed text. `/undo`, sent by an admin as a reply to a saved answer, removes that answer's solution and cascaded feedback; it sends one result message. `/forget` issues an in-memory, one-use token valid for 5 minutes; the same admin must confirm. Deleting that chat's solutions cascades to feedback and clears its pending reply cache. The chat row and threshold remain.

## Data and model contracts

SQLite has three tables defined in `database/schema.py`: `chats` (chat ID, title, threshold, default `0.88`), `solutions` (saved question–answer pair, links, `question_embedding` and `answer_embedding` BLOBs), and `feedback` (solution, query message, user and vote). The solution repository pages results with joined vote counts and updates text within the current chat. Foreign keys are enabled by `database/connection.py`. Schema version 1 is the original layout; version 2 adds the nullable answer vector, filled on first search. `PRAGMA user_version` adopts validated unversioned databases and tracks transactional migrations. `database/backup.py` creates and verifies consistent SQLite copies while polling runs; restoration requires stopping polling. The database path defaults to `data/already_mentioned.db`; Docker uses `/app/data/already_mentioned.db` in a named volume.

The local FastEmbed model is `intfloat/multilingual-e5-small` (384 dimensions). It loads on first use, caches files in root `models/` (Docker: `/app/models`), and encodes in a worker thread. Its optional `threads` argument limits inference CPU concurrency; bot calls keep FastEmbed's default. Saved questions and answers use separate `passage:` vectors; new questions use `query:`. Vectors are normalized and persisted as `float32` bytes. Changing any part of this contract requires recalculating persisted vectors. `diagnostics.py` must use the same embedding rules as the bot.

## Developer entry points

- Dependencies, Python version and CLI entry point: `pyproject.toml`.
- Local and Docker launch: `README.md`, `Dockerfile`, `docker-compose.yml`, Windows `start-docker.cmd`; token template: `.env.example`.
- Targeted automated checks: `tests/test_*.py`; full checks: `python -m ruff check .` and `python -m pytest -q` using Python 3.12 and development dependencies.
- Offline quality and ranking timing check: `python -m already_mentioned.evaluation`; it uses fixed fictional cases, FastEmbed, and no Telegram or SQLite access.
- Historical CSV scoring experiment: `python -m already_mentioned.historical_evaluation --cases PATH --banks PATH...`; checks earlier same-chat solutions and compares question-only, answer-only and max-of-both scoring, with optional separation from the runner-up. Thresholds and margins are calibrated on the earlier 60% of cases per chat and evaluated on the later cases. It calls the bot's embedding service; JSON reports contain hashes and case IDs, not message texts. No bot or database access.
- Local historical bank reviews preserve prior snapshots, source-question context and author provenance. Confirmed administrator additions and inferred organizer additions use separate matching case/bank versions. Unpaired announcements and ambiguous answers remain outside the solution bank; reviewed exports do not import solutions into SQLite.
- Cached historical model comparison: `python -m already_mentioned.historical_model_comparison --model NAME --datasets FOLDER...`; each folder supplies its matching cases and chat banks. One encoder per process, CPU threads limited to two by default, offline cache only. Each scenario selects its threshold on earlier 60% per chat using full F1; later cases, filter misses, warm single-query latency, ranking at 15–200 solutions and process memory are reported separately. No production model, vectors or SQLite changes.
- The committed baseline for the October 3–4 comparison is `docs/experiments/history_models_20261004.json`: aggregate metrics, contracts and input/report hashes without source texts, case IDs or vectors. `docs/HISTORICAL_MODEL_COMPARISON.ru.md` explains the results; `docs/NEXT_EXPERIMENT.ru.md` specifies the pending FRIDA quantization and resource tests.
- Historical top-k reranking: `python -m already_mentioned.reranking_evaluation --cases PATH --banks PATH...`; E5 retrieves candidates before its threshold, then a pinned 8-bit mMARCO MiniLM ONNX cross-encoder scores answer or question-plus-answer pairs on CPU. Both models use two CPU threads by default. Thresholds use earlier cases only. Reports measure candidate coverage, quality, warm latency and process memory; a local CSV queues human review without modifying labels. `--prepare-review-from` prepares review without models using a matching historical report, validating input hashes, embedding contract and case coverage. `--candidates-from --top-k 1` reuses only that report's leaders and measures the reranker alone. The optional `rerank` dependency group supplies inference and memory-measurement libraries.
- Optional model comparison: `python -m already_mentioned.model_comparison`; it uses the same fictional cases, separate embedding spaces, and no production data. The bot model is unchanged.
- Public-data training experiment: `python -m already_mentioned.offline_training`; it learns a diagonal adapter for frozen E5 from a pinned ru-HNP revision and checks held-out data. Outputs go to ignored `training_runs/`; the bot does not load them.
- Public held-out model comparison: `python -m already_mentioned.public_model_comparison`; compares E5, MiniLM and POTION on identical pinned ru-HNP rows, selects each threshold on `val`, and records held-out ranking/classification quality and warm encoding time.
- Extended candidate comparison: `python -m already_mentioned.candidate_model_comparison --model NAME`; runs one of ten Sentence Transformers models at a time on the same pinned ru-HNP rows, selecting the threshold on `val` and measuring CPU query latency. Optional `benchmark` and `benchmark-qwen` dependency groups use different Transformers versions and are isolated from the bot.
- More detailed file inventory, setup, customization and limitations: [docs/DEVELOPMENT.ru.md](docs/DEVELOPMENT.ru.md).

Keep this map synchronized with code whenever the flow, component ownership, storage or model contract changes; `AGENTS.md` contains the update rule for every program change.
