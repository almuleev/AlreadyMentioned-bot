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
| Search | `services/questions.py`, `embeddings.py`, `similarity.py`, `search.py` | Select likely Russian questions, encode them, compare with saved questions from the same chat, apply the chat threshold. |
| Temporary state | `services/reply_cache.py`, `confirmations.py` | Hold reply chains for optional `/solve` and `/forget` confirmations in memory; state is lost on restart. |
| Persistence | `repositories/chats.py`, `solutions.py`, `feedback.py`, `database/connection.py`, `schema.py`, `models/entities.py` | SQLite access, schema, and returned dataclasses. `models/entities.py` is not the ML model. |
| Diagnostics | `diagnostics.py` | Read-only CLI similarity check for one supplied question. |

## Main flows

**Save a solution:** A group member posts a text question → an administrator replies with a text answer → `messages.py` calls `solve.remember_admin_answer()` and stores the candidate in `ReplyCache` → for a question passing `questions.py` and with two direct links, `autosave_admin_answer()` encodes the original question and persists the pair silently. The administrator can optionally reply to their own answer with `/solve` within 6 hours to save a question rejected by the filter. Direct message links make supergroups the practical target. The answer text is not embedded.

**Suggest an answer:** `messages.py` ignores private chats, bots, commands and non-text messages → `questions.py` filters likely questions → `SearchService` loads saved solutions of the current chat only → `EmbeddingService.embed_query()` encodes the new question → `similarity.py` compares it with each saved question vector → the highest score is returned only when it meets the chat threshold → the handler replies to the new question with the saved answer as an escaped HTML blockquote, feedback buttons, and a source-link button. The preview is capped at 3000 characters. This is a linear scan of the chat's solutions. Feedback is stored, not used for ranking.

**Manage chat data:** `/status` reads count and threshold; `/threshold` changes that chat's threshold after an admin check. `/undo`, sent by an admin as a reply to a saved answer, removes that answer's solution and cascaded feedback; it sends one result message. `/forget` issues an in-memory, one-use token valid for 5 minutes; the same admin must confirm. Deleting that chat's solutions cascades to feedback and clears its pending reply cache. The chat row and threshold remain.

## Data and model contracts

SQLite has three tables defined in `database/schema.py`: `chats` (chat ID, title, threshold, default `0.88`), `solutions` (saved question–answer pair, links and `question_embedding` BLOB), and `feedback` (solution, query message, user and vote). Foreign keys are enabled by `database/connection.py`. Schema initialization creates missing tables but does not migrate existing ones. The database path defaults to `data/already_mentioned.db`; Docker uses `/app/data/already_mentioned.db` in a named volume.

The local FastEmbed model is `intfloat/multilingual-e5-small` (384 dimensions). It loads on first use, caches files in root `models/` (Docker: `/app/models`), and encodes in a worker thread. Saved questions use `passage:`, new questions use `query:`; vectors are normalized and persisted as `float32` bytes. Changing any part of this contract requires recalculating persisted vectors. `diagnostics.py` must use the same embedding rules as the bot.

## Developer entry points

- Dependencies, Python version and CLI entry point: `pyproject.toml`.
- Local and Docker launch: `README.md`, `Dockerfile`, `docker-compose.yml`; token template: `.env.example`.
- Targeted automated checks: `tests/test_*.py`; full checks: `python -m ruff check .` and `python -m pytest -q` using Python 3.12 and development dependencies.
- More detailed file inventory, setup, customization and limitations: [docs/DEVELOPMENT.ru.md](docs/DEVELOPMENT.ru.md).

Keep this map synchronized with code whenever the flow, component ownership, storage or model contract changes; `AGENTS.md` contains the update rule for every program change.
