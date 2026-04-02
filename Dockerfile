FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --verbose
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

COPY ./src/api ./src/api
COPY ./src/lib ./src/lib
COPY ./src/config.py ./src/config.py

ARG CACHEBUST=1
COPY ./src/cache ./src/cache

EXPOSE 8000

CMD ["uv", "run", "python", "-u", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0"]