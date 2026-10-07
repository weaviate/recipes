# Drop a Vector Index

A recipe demonstrating how to **drop the index of a named vector** from an existing collection (GA in Weaviate v1.40). This reclaims the disk space of an embedding you no longer use, without recreating the collection or re-ingesting your data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weaviate/recipes/blob/main/weaviate-features/drop-vector-index/drop-vector-index.ipynb)

📓 [`drop-vector-index.ipynb`](./drop-vector-index.ipynb)

## What it covers

| Section | API |
|---|---|
| Collection with two named vectors | `Configure.Vectors.text2vec_openai(...)`, `Configure.Vectors.text2vec_cohere(...)` |
| Measure disk use per vector index | shard folder on disk |
| Drop the index | `DELETE /v1/schema/{collection}/vectors/{vector}/index` |
| Wait for the asynchronous drop to finish | `GET /v1/schema/{collection}` |
| Verify objects, remaining vectors and search | `query.near_vector(target_vector=...)` |

In the captured run, dropping a 3072-dimension OpenAI vector from 2,000 passages reclaimed **58% of the shard's disk** in about 35 seconds.

## Requirements

| | Minimum |
|---|---|
| Weaviate | `v1.40` (outputs captured on `v1.40.0`) |
| `weaviate-client` (Python) | `v4.23.1` |

The Python client release after `4.23.1` adds `collection.config.delete_vector_index()`. This notebook calls the REST endpoint directly so it works with the current release.

## Running it

The recipe needs **no API keys**: it imports vectors that were already computed and searches with `near_vector`. It runs [Embedded Weaviate](https://docs.weaviate.io/deploy/installation-guides/embedded), so it also runs in Colab.

```bash
python -m venv .venv
. .venv/bin/activate
pip install "weaviate-client>=4.23.1" datasets jupyter
jupyter notebook drop-vector-index.ipynb
```

## Data

2,000 passages from [`weaviate/wiki-sample`](https://huggingface.co/datasets/weaviate/wiki-sample), loaded twice: pre-vectorized with OpenAI `text-embedding-3-large` (the vector we drop) and with Cohere `embed-multilingual-v3.0` (the vector we keep).
