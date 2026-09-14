# Server-Side Batching

A recipe demonstrating [server-side batching](https://docs.weaviate.io/weaviate/manage-objects/import#server-side-batching) — Weaviate's import mode where the **server** tells the client how fast to send data, instead of you tuning a `batch_size` by hand.

📓 [`server_side_batching.ipynb`](./server_side_batching.ipynb)

## What it covers

| Section | API |
|---|---|
| Import a list already in memory | `collection.data.ingest(objects)` |
| Stream a large file without loading it | `collection.data.ingest(generator)` |
| Per-object control and mid-import error checks | `with collection.batch.stream() as batch:` |
| Reading failures | `BatchObjectReturn.errors`, `collection.batch.failed_objects` |
| Benchmark vs. client-side batching | `batch.fixed_size()`, `batch.dynamic()` |

It closes with a benchmark of all six import styles over the same 20,000 objects, and an honest reading of the result — on an idle local instance with pre-computed vectors, a hand-tuned `fixed_size` can beat server-side batching. The notebook explains why that happens and why it inverts under real load.

## Requirements

| | Minimum |
|---|---|
| Weaviate | `v1.36` (this recipe's outputs were captured on `v1.39.4`) |
| `weaviate-client` (Python) | `v4.20.0` |

Server-side batching uses the gRPC API, which current clients enable automatically.

> Also available in the TypeScript, Java and C# clients. The Go client does not support it — use manual batching there.

## Running it

The recipe needs **no API keys and no inference provider**: the collection is created with `Configure.Vectors.self_provided()` and vectors come from a small hashing embedder defined in the notebook. That keeps the focus on the batching API rather than on any one model provider.

Start a local Weaviate (runs in the foreground — leave it in its own terminal):

```bash
docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.39.4
```

Then:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
jupyter notebook server_side_batching.ipynb
```

To run against Weaviate Cloud or another `1.36+` deployment instead, swap the `connect_to_local()` call for `connect_to_weaviate_cloud()`. Everything else works unchanged.

The notebook deletes the collections it creates and removes its temporary JSONL file in the final cell.

## Data

[`datasets/1k_products.csv`](../../../datasets/1k_products.csv) from this repo, fetched over HTTPS so the notebook also runs standalone in Colab. The streaming section expands it into a 20,000-row JSONL file on disk.
