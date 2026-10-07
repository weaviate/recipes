# The Weaviate Search REST API

A recipe demonstrating Weaviate's **dedicated REST API for search** (on by default since v1.40): one `POST` endpoint per search type, a JSON body in, and a consistent JSON envelope out, with no GraphQL or gRPC.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weaviate/recipes/blob/main/weaviate-features/search-rest-api/search-rest-api.ipynb)

📓 [`search-rest-api.ipynb`](./search-rest-api.ipynb)

## What it covers

| Section | Endpoint |
|---|---|
| Keyword search with property boosts | `POST /v1/search/{collection}/bm25` |
| Vector search from text, including a multilingual query | `POST /v1/search/{collection}/near-text` |
| Vector search from your own vector | `POST /v1/search/{collection}/near-vector` |
| Vector search from a stored object | `POST /v1/search/{collection}/near-object` |
| Keyword + vector fusion with `alpha` | `POST /v1/search/{collection}/hybrid` |
| `where`, `returnProperties`, `returnReferences`, `returnMetadata` | shared by all search endpoints |
| Total, filtered and grouped counts | `POST /v1/aggregate/{collection}` |
| The same search from `curl`, and with the Python client | |

## Requirements

| | Minimum |
|---|---|
| Weaviate | `v1.40` (outputs captured on `v1.40.0`) |
| `weaviate-client` (Python) | `v4.23.1`, used for setup only |
| Cohere API key | for query vectorization (`near-text`, `hybrid`) |

## Running it

The recipe runs [Embedded Weaviate](https://docs.weaviate.io/deploy/installation-guides/embedded), so it also runs in Colab. Set `COHERE_APIKEY` in your environment, or paste the key when the notebook asks for it.

```bash
python -m venv .venv
. .venv/bin/activate
pip install "weaviate-client>=4.23.1" requests datasets jupyter
jupyter notebook search-rest-api.ipynb
```

## Data

2,000 passages from [`weaviate/wiki-sample`](https://huggingface.co/datasets/weaviate/wiki-sample), **pre-vectorized** with Cohere `embed-multilingual-v3.0`. Importing them makes no calls to Cohere. The same dataset is also available with OpenAI `text-embedding-3-small` and `text-embedding-3-large` vectors.
