# Instructions for agents working in this repository

## Read first, then read only what the task needs

This is a Python 3.12 Telegram bot (aiogram 3, SQLite, Sentence Transformers). Start with [ARCHITECTURE.md](ARCHITECTURE.md) for the component map and data flows. Use [docs/DEVELOPMENT.ru.md](docs/DEVELOPMENT.ru.md) when you need detailed setup, file responsibilities, or extension guidance. [docs/GUIDE.ru.md](docs/GUIDE.ru.md) describes user-visible behavior. Avoid scanning the whole tree by default: select the relevant files from the map below and their tests.

| Task area | Read first | Relevant tests |
| --- | --- | --- |
| Startup, configuration, Docker | `src/already_mentioned/main.py`, `config.py`, `bot/factory.py`, `pyproject.toml`, `Dockerfile`, `docker-compose.yml` | `test_config.py` |
| Commands and Telegram interactions | `src/already_mentioned/handlers/`, `bot/permissions.py` | `test_solve.py`, `test_management.py`, `test_messages.py` |
| Question detection and ranking | `src/already_mentioned/services/questions.py`, `search.py`, `similarity.py`, `evaluation.py`, `evaluation_cases.py`, `historical_evaluation.py`, `threshold_calibration.py` | `test_search.py`, `test_similarity.py`, `test_evaluation.py`, `test_historical_evaluation.py`, `test_threshold_calibration.py` |
| Embedding model and offline experiments | `src/already_mentioned/services/embeddings.py`, `services/reranking.py`, `reranking_evaluation.py`, `diagnostics.py`, `model_comparison.py`, `historical_model_comparison.py`, `offline_training.py`, `public_model_comparison.py`, `candidate_model_comparison.py` | `test_embeddings.py`, `test_reranking.py`, `test_reranking_evaluation.py`, `test_diagnostics.py`, `test_evaluation.py`, `test_historical_model_comparison.py`, `test_offline_training.py`, `test_public_model_comparison.py`, `test_candidate_model_comparison.py` |
| Persistence, migrations, backups | `src/already_mentioned/database/`, `repositories/`, `models/entities.py` | `test_database.py`, `test_schema_migrations.py`, `test_backup.py`, `test_frida_migration.py` |
| Local model-result inspection | `src/already_mentioned/model_results_viewer.py`, `model_results_viewer.html`, `docs/MODEL_RESULTS_VIEWER.ru.md` | `test_model_results_viewer.py` |
| FRIDA search experiments | `src/already_mentioned/search_experiment.py`, `reranker_input_experiment.py`, `services/search.py`, `diagnostics.py`, `docs/SEARCH_IMPROVEMENTS.ru.md`, `docs/RERANKER_INPUTS.ru.md` | `test_search_experiment.py`, `test_reranker_input_experiment.py`, `test_search.py`, `test_diagnostics.py` |
| Production hybrid and release replay | `src/already_mentioned/config.py`, `main.py`, `services/search.py`, `release_validation.py`, `partial_label_review.py`, `docs/ACTIVE_SEARCH.ru.md` | `test_hybrid_search.py`, `test_release_validation.py`, `test_partial_label_review.py`, `test_config.py` |

Do not read or print `.env`, SQLite files in `data/`, or downloaded model files in the root `models/` unless the task specifically needs them. `src/already_mentioned/models/` contains **source dataclasses** and must stay in Git and Docker builds. Existing local data and model caches must be preserved.

User-facing text, the website, and user documentation are Russian-only. Keep program names, commands, paths, and established technical terms as needed. `AGENTS.md` and `ARCHITECTURE.md` may remain in English because they primarily guide agents; do not add a duplicate English user guide.

## Invariants to preserve

- The bot replies to a new question with one saved administrator answer as an escaped HTML quote, plus a source link in the text and one row of 👍/👎 buttons. Only the question author may vote; buttons disappear after a vote or one hour. The SQLite cleanup queue survives restarts. It does not generate an answer or train the model. Search and settings are scoped to the current Telegram chat. Keep the reply within Telegram's message limit; long answers are previewed at 3000 characters.
- An administrator's text reply to a likely question is saved silently when both Telegram message links exist. `/solve` is an optional manual fallback for questions rejected by the filter; its pending chain lives in memory for up to 6 hours. `/undo` as a reply to the saved answer deletes that solution and its votes. Preserve silent autosave to avoid chat spam.
- Production uses pinned FP32 `ai-forever/FRIDA` (1536d, CLS, 512 tokens, L2, float32-le): saved questions/answers use `search_document:`, new questions use `search_query:`. E5 remains only in explicitly legacy offline experiments. Bot and diagnostics require matching SQLite contract and complete valid vectors; never mix encoders. `database/reembed.py` backs up before full transactional re-embedding with the bot stopped; polling and migration share a process lock.
- `offline_training.py` only trains and evaluates an experimental adapter on public data. Its output is ignored by the bot and must not be used with stored vectors without an explicit migration and full re-embedding.
- `historical_evaluation.py` reads explicitly supplied labelled CSV exports only. Its question/answer scoring and margin experiments are offline; calibrate on earlier cases and report later cases separately. Unknown labels are excluded, and expected solutions must precede each query in the same chat. Historical labels remain provisional until independently reviewed.
- Historical bank reviews must retain source IDs, dates and author provenance. Evaluate additions from confirmed administrators separately from inferred organizers; announcements without a paired question are not silently saved solutions. Short answers must retain their source-question context and must not be automatically imported into the working database. `historical_model_comparison.py` compares cached encoders in separate spaces on matching CSV versions, calibrates full F1 on earlier cases and reports later cases; it never changes the bot model or database.
- The offline `frida-int8` variant dynamically quantizes Linear layers of a locally loaded FRIDA copy. Original cached files remain unchanged; all experiment query/question/answer vectors are recomputed. Reports distinguish conversion/load memory, post-query RSS and whole-process peak; post-query samples are not guaranteed within-call peaks.
- Offline `--load-cycles` tests five serial chat workloads with 200 same-chat historical solution pairs per request. It measures queue latency on the host and must not be described as a Telegram/VPS test or an enforced memory limit.
- `reranking_evaluation.py` experiments with a pinned quantized multilingual ONNX cross-encoder on top-k same-chat historical candidates. Pair scores are relevance estimates, not proof of answer completeness. It does not change bot search, stored vectors or labels; its review CSV contains source text and must remain local. `--prepare-review-from` builds human review from a hash- and contract-matched historical report without loading models or claiming new metrics. `--candidates-from --top-k 1` scores that report's single candidates without loading E5; this cannot test alternative leaders or joint model memory.
- `feedback` records votes but does not affect ranking. `/forget` removes one chat's solutions and their votes, while retaining the chat and threshold.
- New chats receive the provisional FRIDA threshold 0.458 explicitly from `config.py`; the legacy SQL default remains 0.88. Refreshing a chat never overwrites its threshold. Calibration uses only early main cases; beta 1.1 reflects a slight recall preference. Current late precision is poor; do not describe this setting as independently validated.
- `/solutions` lists only the current chat for administrators. `/editquestion` and `/editanswer` require a matching solution ID and reply to its saved answer; each edit recomputes the document embedding of the changed text and keeps the other vector.
- SQLite schema version 1 is the original three-table layout; version 2 adds the nullable answer embedding; version 3 records the database-wide embedding contract, adopting old vectors as E5 without rewriting them; version 4 adds the persistent feedback-keyboard cleanup queue. New empty databases adopt FRIDA at startup; populated E5 databases require explicit re-embedding. Schema migrations are transactional. Backups use SQLite's online backup API and must be verified before offline restoration; E5 rollback also requires the prior E5 code version.
- A token must have only one running long-polling process. Never commit or expose `TELEGRAM_BOT_TOKEN`.

The preceding 0.458 / beta-1.1 settings describe the preserved baseline and historical experiments. The current production hybrid follows the partial-answer / F2 policy below and has its own threshold. Schema 5 and the production MiniLM path below supersede the old offline-only reranking boundary; historical experiment CLIs remain offline.

## Change workflow

Production search supports `SEARCH_MODE=hybrid` (FRIDA max cosine 75% + pinned MiniLM pair score 25%, top five, batch_size=1, 512 tokens) and `baseline` rollback. SQLite version 5 adds nullable per-chat `hybrid_threshold`, separate from legacy `similarity_threshold`; refreshing a chat preserves both. `/threshold` and `/status` use the active mode. Pair inference runs serially off the event loop; cache failure blocks startup rather than silently changing scoring.

`release_validation.py` runs the real SearchService on all explicit export rows, including unknowns for execution only. It fits no thresholds, uses no Telegram or working SQLite, and retrieves only prior same-chat solutions. Report real container limits separately from host experiments and do not call offline replay a live Telegram/VPS test.

The owner now accepts substantive partial answers as correct and prefers recall. `partial_label_review.py` creates separate source-preserving annotations from explicit F/P/N/U reviews; F and P are acceptable, N is not, unknown remains excluded. Reports under this policy use main-early F2 calibration (`--beta 2`), retain frozen chronological splits and inferred-organizer scenarios separately; prior strict-completeness reports remain historical and must not be compared as identical labels.

`search_experiment.py` is offline: keep full FP32 FRIDA, recompute contextual document vectors separately, retrieve only past same-chat solutions, and freeze the source-ID split before excluding unknown labels. Calibrate strategies on main early cases only and report later/proxy results separately. Manual completeness reviews in the local viewer are independent annotations, never automatic label or ranking changes.

`reranker_input_experiment.py` reuses a hash-matched full-FRIDA baseline's five candidates and frozen splits. It compares single-pair ONNX input formats and fixed balanced/hybrid scores; thresholds use main early cases only. It loads no encoder, never expands candidate coverage, and cannot measure complete bot latency or joint model memory. The pinned quantized reranker has observed batch-dependent scores; do not mix calibrated thresholds across batch/input settings.

Its `experimental_transport_filter()` helper is an offline-only hypothesis for implicit transport-delay requests; the production question filter does not call it. Local filter-ablation scripts must encode all newly admitted unscored negatives and report their false prompts, rather than enabling only known positives or claiming unmeasured quality.

The input experiment's optional `--reranker gte-fp32` uses a pinned community FP32 ONNX export with single-pair inference and the same 512-token cap, without loading remote Python code. Record its own model/file/revision and calibrate its own thresholds. `--strategies` restricts inference to needed formats; downloads require the explicit `--download-model` option. Production and stored FRIDA vectors remain unchanged.

Keep edits focused; check the corresponding tests before broadening the search. Run with the project venv when present:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

For changes in behavior, add or update focused tests and update the relevant user documentation. **After every program change, check whether `AGENTS.md` or `ARCHITECTURE.md` has become inaccurate; update the affected file(s) in the same change.** In particular, update them when entry points, folders, dependencies, commands, configuration, data schema, model behavior, component boundaries, or the documented invariants change. Keep these two files concise and factual; put detailed explanations in `docs/DEVELOPMENT.ru.md`.
