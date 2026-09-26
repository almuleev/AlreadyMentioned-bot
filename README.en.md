# AlreadyMentioned

[Русский](README.md) · **English**

AlreadyMentioned is a self-hosted Telegram bot for support chats. It finds a previously confirmed answer and links to it; it does not generate new answers.

**Status: local MVP.** Administrators confirm text question–answer pairs with `/solve`. The bot checks likely Russian questions against confirmed solutions in the same chat and offers one link when the similarity reaches that chat's threshold. The core flow was verified in a Telegram test supergroup.

Step-by-step setup, usage rules, and similarity-score explanations are available in the **[Russian user guide](docs/GUIDE.ru.md)**.

[Project website](https://almuleev.github.io/AlreadyMentioned-bot/) · [GitHub repository](https://github.com/almuleev/AlreadyMentioned-bot)

## Structure

- `src/already_mentioned/bot/` — aiogram bot creation and permission checks.
- `src/already_mentioned/handlers/` — commands, messages, and feedback buttons.
- `src/already_mentioned/services/` — short-lived reply cache, question filter, local embeddings, and similarity search.
- `src/already_mentioned/database/` — SQLite connection and schema initialization.
- `src/already_mentioned/repositories/` — scoped storage operations for chats, solutions, and feedback.
- `src/already_mentioned/models/` — data structures returned by repositories.
- `tests/` — offline tests; no Telegram token or embedding model is needed.

## Local setup (Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Fill `TELEGRAM_BOT_TOKEN` in `.env` locally. `DATABASE_PATH` defaults to `data/already_mentioned.db`. Keep `.env` out of Git. If Windows or environment variables configure an HTTPS proxy, the bot uses it for Telegram. The model is downloaded into the ignored `models/` directory on the first `/solve`, then used locally. Imports and tests do not download it. Embedding work runs outside the event loop.

To use `/solve`, the bot must receive ordinary group messages (disable its Privacy Mode in BotFather). Public and private supergroups have direct message links. Telegram does not provide them for a basic group, so a solution cannot be saved there. The bot remembers eligible administrator replies only in memory for up to six hours or until restart; at most 5,000 reply chains are retained.

Commands: `/start` and `/help` show guidance; `/status` shows the number of solutions and the current threshold; `/threshold 0.88` changes this chat's threshold (administrator only); `/forget` requests confirmation before deleting this chat's solutions and feedback (administrator only). Each user may vote once per suggestion using **👍 Помогло** or **👎 Не подходит**. A new chat starts with a threshold of 0.88.

Start long polling:

```powershell
python -m already_mentioned.main
```

## Docker Compose (optional local deployment)

Docker Desktop with Linux containers and Compose is required. Keep the bot token in the ignored `.env` file. Stop any Python polling process before starting Compose: only one process may poll the same bot token.

```powershell
docker compose build
docker compose up -d
docker compose logs -f bot
```

`docker compose down` stops the container. The named volumes `bot_data` and `model_cache` preserve the SQLite database and model between runs. **Do not use `down --volumes` if you want to keep the data.** Docker uses a separate database and cache: files in the Windows `data/` and `models/` folders are not imported automatically. No network port is exposed.

To inspect a question's numerical similarity score, run `docker compose exec -T bot python -m already_mentioned.diagnostics "Your question here?"`. See the [Russian guide](docs/GUIDE.ru.md#как-посмотреть-значение-самому) for details.

If Telegram is reachable only through a proxy, set `HTTPS_PROXY` in `.env` to an address the container can reach. A proxy listening on Windows `127.0.0.1` needs a container-reachable host address, such as `host.docker.internal` with the same port. The proxy address is passed to the bot container and should be kept out of Git.

## Checks and privacy

Checks do not require a token:

```powershell
ruff check .
pytest
```

The bot stores only administrator-confirmed question–solution pairs, feedback, and minimal chat settings, not the full chat history or user names. Tables are created on startup. `/forget` deletes a chat's solutions and feedback only after an administrator presses the confirmation button; the chat's threshold setting remains. `tests/fixtures/questions.py` contains 80 synthetic messages for offline checks; these tests use fake vectors and do not measure the real model's accuracy.
