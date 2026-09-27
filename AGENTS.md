# Instructions for agents working in this repository

## Read first, then read only what the task needs

This is a Python 3.12 Telegram bot (aiogram 3, SQLite, FastEmbed). Start with [ARCHITECTURE.md](ARCHITECTURE.md) for the component map and data flows. Use [docs/DEVELOPMENT.ru.md](docs/DEVELOPMENT.ru.md) when you need detailed setup, file responsibilities, or extension guidance. [docs/GUIDE.ru.md](docs/GUIDE.ru.md) describes user-visible behavior. Avoid scanning the whole tree by default: select the relevant files from the map below and their tests.

| Task area | Read first | Relevant tests |
| --- | --- | --- |
| Startup, configuration, Docker | `src/already_mentioned/main.py`, `config.py`, `bot/factory.py`, `pyproject.toml`, `Dockerfile`, `docker-compose.yml` | `test_config.py` |
| Commands and Telegram interactions | `src/already_mentioned/handlers/`, `bot/permissions.py` | `test_solve.py`, `test_management.py`, `test_messages.py` |
| Question detection and ranking | `src/already_mentioned/services/questions.py`, `search.py`, `similarity.py` | `test_search.py`, `test_similarity.py` |
| Embedding model | `src/already_mentioned/services/embeddings.py`, `diagnostics.py` | `test_embeddings.py`, `test_diagnostics.py` |
| Persistence | `src/already_mentioned/database/`, `repositories/`, `models/entities.py` | `test_database.py` |

Do not read or print `.env`, SQLite files in `data/`, or downloaded model files in the root `models/` unless the task specifically needs them. `src/already_mentioned/models/` contains **source dataclasses** and must stay in Git and Docker builds. Existing local data and model caches must be preserved.

User-facing text, the website, and user documentation are Russian-only. Keep program names, commands, paths, and established technical terms as needed. `AGENTS.md` and `ARCHITECTURE.md` may remain in English because they primarily guide agents; do not add a duplicate English user guide.

## Invariants to preserve

- The bot replies to a new question with one saved administrator answer as an escaped HTML quote, plus a source-link button and feedback buttons. It does not generate an answer or train the model. Search and settings are scoped to the current Telegram chat. Keep the reply within Telegram's message limit; long answers are previewed at 3000 characters.
- An administrator's text reply to a likely question is saved silently when both Telegram message links exist. `/solve` is an optional manual fallback for questions rejected by the filter; its pending chain lives in memory for up to 6 hours. `/undo` as a reply to the saved answer deletes that solution and its votes. Preserve silent autosave to avoid chat spam.
- The stored embedding represents the **question**, with the E5 `passage:` prefix. New questions use `query:`. A change of model, prefixes, dimension, normalization, or vector format requires a plan to recompute stored embeddings; do not mix incompatible vectors.
- `feedback` records votes but does not affect ranking. `/forget` removes one chat's solutions and their votes, while retaining the chat and threshold.
- Database schema creation is not a migration system. Changes to existing tables need an explicit migration path and tests.
- A token must have only one running long-polling process. Never commit or expose `TELEGRAM_BOT_TOKEN`.

## Change workflow

Keep edits focused; check the corresponding tests before broadening the search. Run with the project venv when present:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

For changes in behavior, add or update focused tests and update the relevant user documentation. **After every program change, check whether `AGENTS.md` or `ARCHITECTURE.md` has become inaccurate; update the affected file(s) in the same change.** In particular, update them when entry points, folders, dependencies, commands, configuration, data schema, model behavior, component boundaries, or the documented invariants change. Keep these two files concise and factual; put detailed explanations in `docs/DEVELOPMENT.ru.md`.
