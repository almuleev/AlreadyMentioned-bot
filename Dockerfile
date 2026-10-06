FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir . \
    && groupadd --system bot \
    && useradd --system --create-home --gid bot --home-dir /home/bot bot \
    && mkdir -p /app/data /app/models \
    && chown -R bot:bot /app/data /app/models

USER bot

CMD ["python", "-m", "already_mentioned.main"]
