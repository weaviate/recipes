---
layout: recipe
toc: True
title: "Batch ingest into Weaviate with dlt"
featured: False
integration: True
agent: False
tags: ['dlt', 'ETL', 'Batch Import', 'Multi-Tenancy', 'Cohere', 'Integration']
---
[dlt](https://dlthub.com/) is an open-source Python library that extracts data from APIs,
databases and files, normalizes it into a typed schema, and loads it into a destination.
Weaviate is one of those destinations.

This recipe loads **10,000 Wikipedia paragraphs that are already embedded with Cohere** into
Weaviate. The vectors ship with the dataset, so nothing is embedded during the load — the
recipe is about the ingest path itself:

1. read a pre-vectorized Parquet slice,
2. let dlt infer the schema and create the collection,
3. send the objects **and their vectors** with Weaviate's server-side batching,
4. search them with a Cohere-embedded query,
5. give each language its own tenant,
6. re-run so nothing is duplicated.

**Dataset:** [`CohereLabs/wikipedia-2023-11-embed-multilingual-v3`](https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3)
— Wikipedia chunked into paragraphs and embedded with Cohere `embed-multilingual-v3.0`
(1024 dimensions). It is public, so no Hugging Face token is needed.

**What you need:** a Weaviate instance (local Docker, self-hosted, or Weaviate Cloud) and a
[Cohere API key](https://dashboard.cohere.com/api-keys). The key is only used at **query** time:
the collection declares `text2vec-cohere` with the same model the dataset was embedded with, so
Weaviate embeds your search text for you. Loading makes no Cohere calls at all, because every
object already carries its vector.

## Install

`fsspec[http]` rather than plain `fsspec`: the HTTP filesystem needs `aiohttp`, which the bare
package does not pull in, and the Parquet slice is read straight over HTTPS.

```python
%pip install -q "dlt[weaviate]>=1.30.0" "pyarrow>=16" "fsspec[http]"
```

## Where to load

dlt reaches Weaviate three ways and infers which one you mean from the URL. Set `TARGET` below.

| `TARGET` | Use when |
| --- | --- |
| `local` | Weaviate in Docker on this machine |
| `custom` | Self-hosted Weaviate on your own host and ports |
| `cloud` | [Weaviate Cloud](https://console.weaviate.cloud/) |

### Running locally

A `docker-compose.yml` sits next to this notebook. Because the vectors come from the dataset, no
vectorizer module is configured and it is a single container:

```sh
docker compose up -d
```

Two things to know about self-hosting:

- **Expose port 50051.** The v4 Python client needs gRPC; REST alone is not enough.
- **`custom` requires both ports explicitly**, and supports a separate `grpc_host` when REST and
  gRPC are reachable under different names.

```python
import getpass
import os

from dlt.destinations import weaviate

# "local" | "custom" | "cloud"
TARGET = "local"

# the model the dataset was embedded with; the collection declares the same one
COHERE_MODEL = "embed-multilingual-v3.0"
COHERE_API_KEY = os.environ.get("COHERE_API_KEY") or getpass.getpass("Cohere API key: ")

if TARGET == "local":
    credentials = {"url": "http://localhost:8080"}
elif TARGET == "custom":
    credentials = {
        "url": "http://weaviate.internal",
        "http_port": 8080,
        "grpc_port": 50051,
        # "grpc_host": "grpc.internal",  # when gRPC answers under another name
    }
else:
    credentials = {
        "url": input("Cluster REST endpoint: ").strip(),
        "api_key": getpass.getpass("Weaviate API key: "),
    }

# Weaviate calls Cohere with this header when it has to embed a query
credentials["additional_headers"] = {"X-Cohere-Api-Key": COHERE_API_KEY}

def weaviate_destination(**overrides):
    return weaviate(
        credentials=credentials,
        connection_type=TARGET,
        vectorizer="text2vec-cohere",
        module_config={"text2vec-cohere": {"model": COHERE_MODEL}},
        **overrides,
    )

destination = weaviate_destination()
```

```python
# confirm dlt can reach the cluster before loading anything
import dlt

with dlt.pipeline(
    pipeline_name="weaviate_connection_check", destination=destination
).destination_client() as client:
    print("connection type:", client.config.resolve_connection_type())
    print("server version :", client.server_version())
    print("batch mode     :", client.resolve_batch_mode())
```

`batch mode` says how dlt will send the objects. On Weaviate **1.36+** it resolves to `stream`,
which is [server-side batching](https://docs.weaviate.io/weaviate/tutorials/import#option-a-server-side-batching):
the server picks the batch size, the parallelism and the backpressure, so there is nothing to
tune. On older servers dlt falls back to client-side `fixed_size` batching and logs that it did.

## Read the pre-vectorized data

The dataset is published as Parquet. One row group holds exactly 10,000 rows, so the recipe reads
a single row group and never downloads the whole 212 MB file.

`emb` is a 1024-float Cohere vector per paragraph, passed through untouched — the pipeline does no
embedding at all.

```python
import fsspec
import pyarrow.parquet as pq

DATASET = "CohereLabs/wikipedia-2023-11-embed-multilingual-v3"
PARQUET = "https://huggingface.co/api/datasets/{ds}/parquet/{lang}/train/0.parquet"

def wikipedia_rows(wiki_lang: str = "simple", row_group: int = 0):
    parquet_file = pq.ParquetFile(fsspec.open(PARQUET.format(ds=DATASET, lang=wiki_lang)).open())
    for row in parquet_file.read_row_group(row_group).to_pylist():
        yield {
            "doc_id": row["_id"],
            "title": row["title"],
            "text": row["text"],
            "url": row["url"],
            "lang": wiki_lang,
            "embedding": row["emb"],
        }

sample = next(wikipedia_rows())
print(sample["title"], "|", sample["text"][:80])
print("vector dimensions:", len(sample["embedding"]))
```

## Define the resource

`primary_key` plus `write_disposition="merge"` means a re-run updates a paragraph in place
instead of duplicating it.

Two adapter hints do the work here:

- `vector="embedding"` marks `embedding` as the **object vector** rather than a property, so dlt
  sends the vector that shipped with the dataset instead of asking Weaviate to compute one.
- `vectorize="text"` tells the collection which property `text2vec-cohere` should embed. Nothing
  is embedded during this load — a supplied vector always wins — but declaring it means Weaviate
  can embed a *query* later, and that anything you load afterwards without a vector is embedded
  consistently from the same property.

```python
from dlt.destinations.adapters import weaviate_adapter

# NOTE: dlt injects config into resource arguments, and an argument named `lang` would be
# resolved from the LANG environment variable. `wiki_lang` avoids that collision.
@dlt.resource(name="paragraphs", primary_key="doc_id", write_disposition="merge")
def paragraphs(wiki_lang: str = "simple"):
    yield from wikipedia_rows(wiki_lang)

articles = weaviate_adapter(paragraphs(), vectorize="text", vector="embedding")
```

## Load

No batching loop, no retry handling, no collection definition — dlt infers the schema, creates the
collection, and streams the objects through Weaviate's batch API.

```python
import time

pipeline = dlt.pipeline(
    pipeline_name="weaviate_wikipedia",
    destination=destination,
    dataset_name="Wikipedia",
)

started = time.time()
load_info = pipeline.run(articles)
print(load_info)
print(f"\nwall clock: {time.time() - started:.1f}s")
```

```python
with pipeline.destination_client() as client:
    collection = client.db_client.collections.get("Wikipedia_Paragraphs")
    print("objects:", collection.aggregate.over_all(total_count=True).total_count)

    obj = collection.query.fetch_objects(limit=1, include_vector=True).objects[0]
    print("stored vector dimensions:", len(obj.vector["default"]))
    print("properties:", sorted(obj.properties))
```

Note that `embedding` is **not** in the property list. It became the object's vector, so the
floats live in the vector index rather than being duplicated as a property.

## Search

Because the collection declares `text2vec-cohere` with the same model the dataset was embedded
with, `near_text` works: Weaviate sends your query to Cohere using the `X-Cohere-Api-Key` header
and compares the result against the vectors you loaded. There is no Cohere code in this notebook.

This is the reason to declare the vectorizer even though every object arrived with a vector — it
is what makes the collection queryable by text.

```python
def search(query: str, limit: int = 5, **kwargs):
    with pipeline.destination_client() as client:
        collection = client.db_client.collections.get("Wikipedia_Paragraphs")
        hits = collection.query.near_text(
            query=query,
            limit=limit,
            return_properties=["title", "text", "url"],
            **kwargs,
        ).objects
    for hit in hits:
        print(f"[{hit.properties['title']}] {hit.properties['text'][:140]}...")
    return hits

_ = search("how do birds fly?")
```

Wikipedia is chunked per paragraph, so several hits often share a title — they are different
paragraphs of the same article. That is why the snippet is printed and not just the title.

Filters combine with vector search, and they run on the properties dlt inferred. This slice is
the first alphabetical chunk of Simple Wikipedia, so `Chess` is one of the ~889 articles in it:

```python
from weaviate.classes.query import Filter

_ = search(
    "opening moves and strategy",
    limit=3,
    filters=Filter.by_property("title").equal("Chess"),
)
```

## One tenant per language

Weaviate [multi-tenancy](https://docs.weaviate.io/weaviate/manage-collections/multi-tenancy)
gives each tenant its own shard, which is how you isolate customers — or, here, languages. dlt
takes `multi_tenancy` and `tenant` on the destination.

Two things are on by default and worth knowing about:

- `auto_tenant_creation` — the tenant is created the first time the pipeline writes to it, so you
  do not have to provision tenants up front.
- `auto_tenant_activation` — an inactive tenant is activated by any operation against it. Without
  it, loading into a tenant that had been deactivated fails with `tenant not active`.

Cohere's `embed-multilingual-v3.0` puts every language in one vector space, so an English query
still matches German paragraphs.

dlt's own `_dlt_*` bookkeeping collections stay single-tenant: they hold pipeline state, which is
already keyed by pipeline name, not tenant data.

```python
LANGUAGES = ["simple", "de"]

for lang in LANGUAGES:
    tenant_pipeline = dlt.pipeline(
        pipeline_name=f"weaviate_wikipedia_{lang}",
        destination=weaviate_destination(multi_tenancy=True, tenant=lang),
        dataset_name="WikipediaByLang",
    )
    tenant_pipeline.run(
        weaviate_adapter(paragraphs(lang), vectorize="text", vector="embedding")
    )
    print(f"{lang}: loaded")
```

```python
with tenant_pipeline.destination_client() as client:
    collection = client.db_client.collections.get("WikipediaByLang_Paragraphs")
    tenancy = collection.config.get().multi_tenancy_config
    print("multi-tenant     :", tenancy.enabled)
    print("auto creation    :", tenancy.auto_tenant_creation)
    print("auto activation  :", tenancy.auto_tenant_activation)

    for lang in LANGUAGES:
        scoped = collection.with_tenant(lang)
        total = scoped.aggregate.over_all(total_count=True).total_count
        top = scoped.query.near_text(
            query="the history of the Roman Empire",
            limit=1,
            return_properties=["title", "text"],
        ).objects[0]
        print(f"\n{lang}: {total} paragraphs")
        print(f"   [{top.properties['title']}] {top.properties['text'][:110]}...")
```

## Re-run it

Running the same pipeline again does not duplicate anything: `merge` on `doc_id` updates the
paragraphs that are already there. Point it at a different row group and only the new paragraphs
are added.

```python
load_info = pipeline.run(weaviate_adapter(paragraphs(), vectorize="text", vector="embedding"))
print(load_info)

with pipeline.destination_client() as client:
    collection = client.db_client.collections.get("Wikipedia_Paragraphs")
    print(
        "objects after re-loading the same slice:",
        collection.aggregate.over_all(total_count=True).total_count,
    )
```

## Where to go next

- **Tune the batching.** `batch_mode` also takes `fixed_size` (with `batch_size` and
  `batch_workers`) and `rate_limit` (with `batch_requests_per_minute`), which is what you want
  when an embedding provider rate-limits you.
- **Named vectors.** `weaviate_adapter(data, named_vectors={...})` gives one collection several
  independently configured vectors.
- **Other sources.** The only Weaviate-specific code above was `weaviate_adapter` and the
  destination name. Point dlt at a database, at S3, or at one of its
  [verified sources](https://dlthub.com/docs/dlt-ecosystem/verified-sources/) and the rest of the
  pipeline is unchanged.
- **Reference:** the [dlt Weaviate destination docs](https://dlthub.com/docs/dlt-ecosystem/destinations/weaviate)
  cover batching, multi-tenancy, named vectors and the connection options.

### Clean up

```python
for p in (pipeline, tenant_pipeline):
    with p.destination_client() as client:
        client.drop_storage()
print("collections dropped")
```