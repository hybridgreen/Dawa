FROM python:3.11

WORKDIR /app

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

COPY pyproject.toml  uv.lock ./

RUN uv sync --frozen
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

ARG CACHEBUST=1

COPY ./src ./src

EXPOSE 8000

CMD ["uv", "run", "python", "-u", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0"]