FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -c "import subprocess,sys,tomllib; p=tomllib.load(open('pyproject.toml','rb'))['project']; subprocess.check_call([sys.executable,'-m','pip','install','--no-cache-dir',*p['dependencies'],*p['optional-dependencies']['rerank']])" \
    && groupadd --system bot \
    && useradd --system --create-home --gid bot --home-dir /home/bot bot \
    && mkdir -p /app/data /app/models \
    && chown -R bot:bot /app/data /app/models

COPY src ./src
COPY README.md ./
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation '.[rerank]'

USER bot

CMD ["python", "-m", "already_mentioned.main"]
