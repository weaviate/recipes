# Boost Recipe

This recipe shows how to use Weaviate's new Boost feature (v1.38+). Boost lets you re-rank vector or keyword search results using filters, numeric/time decay functions, property values, and blends — without changing the underlying retrieval vector model. It is particularly useful for time based queries (such as memories) or boosting vector search
retrieval by numeric values like popularity.

> **Status:** This recipe uses the released Weaviate server 1.38.0 and a preview build of the Python client.
>
> - Python client PR: https://github.com/weaviate/weaviate-python-client/pull/2030

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- `OPENAI_API_KEY` for query and search app

## Quick Start

### 1. Start Weaviate locally

```bash
docker compose up -d
```

### 2. Install dependencies and import data

```bash
uv run import_amazon_products.py
```

This loads Amazon products (with pre-computed OpenAI embeddings) from HuggingFace dataset https://huggingface.co/datasets/milistu/AMAZON-Products-2023

### 3. Run the CLI demo queries

```bash
OPENAI_API_KEY=sk-... uv run query_with_rank.py
```

Runs vector, BM25, and hybrid queries with different boost configurations (filter boost, numeric decay, time decay, property rank, blends) against the imported data and prints results to the terminal.

Options:
- `--depth N` — set rescore depth (default: server default of 100)
- `--limit N` — results per query (default: 10)

### 4. Run the interactive search app

```bash
OPENAI_API_KEY=sk-... uv run search_app.py
```

Open http://localhost:8000 to search products with a web UI. Switch between boost profiles and adjust weight/depth sliders to see how boosting affects results in real time.

## Boost Types

| Type | Description | Example |
|------|-------------|---------|
| **Filter** | Binary boost: objects matching a filter get a score bonus | Promote rating >= 4.9 |
| **Numeric Decay** | Score decays with distance from an origin value | Prefer price near $20 |
| **Time Decay** | Score decays with time distance from a date | Boost products near a date |
| **Property** | Use a numeric property value as the boost score | Rank by popularity |
| **Blend** | Combine multiple boosts into one | Quality + affordability |

## Files

| File | Description |
|------|-------------|
| `import_amazon_products.py` | Load Amazon Products dataset into Weaviate |
| `query_with_rank.py` | CLI demo of all boost types |
| `results.txt` | Example output from `query_with_rank.py` |
| `search_app.py` | Interactive FastAPI web app |
| `docker-compose.yml` | Weaviate 1.38.0 docker compose |
