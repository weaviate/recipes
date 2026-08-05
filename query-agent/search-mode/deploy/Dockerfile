# Stage 1: install dependencies with uv into a self-contained virtualenv
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

COPY app ./app

# Stage 2: slim runtime image without uv or build tooling
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN useradd --create-home appuser
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"

USER appuser
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
