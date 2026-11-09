---
layout: recipe
toc: True
title: "Schema Linking for Text-to-SQL: Vector, Keyword or Hybrid Search?"
featured: False
integration: False
agent: False
tags: ['Hybrid Search', 'Evaluation', 'Text-to-SQL']
---
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weaviate/recipes/blob/main/weaviate-features/evaluation/schema_linking_for_text_to_sql.ipynb)

# Schema Linking for Text-to-SQL: Vector, Keyword or Hybrid Search?

Before a text-to-SQL agent writes a query, it has to find the tables and columns a question is about. This step is called **schema linking**. Semantic layers such as [NVIDIA Auto Ontology](https://github.com/NVIDIA/auto-ontology) store one short text per table and per column, and look them up with vector search.

In this recipe you will:

1. Index two public sample databases in Weaviate, one object per table and one per column.
2. Ask two kinds of questions: ones that **name the column** (`rental_rate for each film`) and **plain-English** ones (`How much does it cost to rent a movie?`).
3. Measure vector, keyword (BM25) and hybrid search on both, then try a simple router that picks a search per question.

Vectors come from a small open embedding model that runs on CPU, so no API key is needed.

*Tested with Weaviate `1.40.4` (Docker) and `1.40.2` (Embedded), Weaviate Python client `4.24.1` and sentence-transformers `6.2.0`.*

*Author: Abhishek Sharma ([@abhisheksharma001](https://github.com/abhisheksharma001))*

## Dependencies

```python
%pip install -q -U weaviate-client sentence-transformers pandas
```

Python output:
```text
Note: you may need to restart the kernel to use updated packages.
```
## Connect to Weaviate

This uses a local Weaviate if one is running (for example via Docker) and otherwise starts [Embedded Weaviate](https://docs.weaviate.io/deploy/installation-guides/embedded), which needs no setup.

```python
import weaviate

try:
    client = weaviate.connect_to_local()
except Exception:
    client = weaviate.connect_to_embedded(environment_variables={"LOG_LEVEL": "error"})

print(client.is_ready())
```

Python output:
```text
True
```
## The schemas

Column names and types from two public sample databases: [Chinook](https://github.com/lerocha/chinook-database) (a music store, MIT License) and [Pagila](https://github.com/devrimgunduz/pagila) (a DVD rental store, PostgreSQL License). The monthly partition copies of Pagila's `payment` table are left out.

```python
SCHEMA = {
    "chinook": {
        "Album": [("AlbumId", "INTEGER"), ("Title", "NVARCHAR(160)"), ("ArtistId", "INTEGER")],
        "Artist": [("ArtistId", "INTEGER"), ("Name", "NVARCHAR(120)")],
        "Customer": [("CustomerId", "INTEGER"), ("FirstName", "NVARCHAR(40)"), ("LastName", "NVARCHAR(20)"), ("Company", "NVARCHAR(80)"), ("Address", "NVARCHAR(70)"), ("City", "NVARCHAR(40)"), ("State", "NVARCHAR(40)"), ("Country", "NVARCHAR(40)"), ("PostalCode", "NVARCHAR(10)"), ("Phone", "NVARCHAR(24)"), ("Fax", "NVARCHAR(24)"), ("Email", "NVARCHAR(60)"), ("SupportRepId", "INTEGER")],
        "Employee": [("EmployeeId", "INTEGER"), ("LastName", "NVARCHAR(20)"), ("FirstName", "NVARCHAR(20)"), ("Title", "NVARCHAR(30)"), ("ReportsTo", "INTEGER"), ("BirthDate", "DATETIME"), ("HireDate", "DATETIME"), ("Address", "NVARCHAR(70)"), ("City", "NVARCHAR(40)"), ("State", "NVARCHAR(40)"), ("Country", "NVARCHAR(40)"), ("PostalCode", "NVARCHAR(10)"), ("Phone", "NVARCHAR(24)"), ("Fax", "NVARCHAR(24)"), ("Email", "NVARCHAR(60)")],
        "Genre": [("GenreId", "INTEGER"), ("Name", "NVARCHAR(120)")],
        "Invoice": [("InvoiceId", "INTEGER"), ("CustomerId", "INTEGER"), ("InvoiceDate", "DATETIME"), ("BillingAddress", "NVARCHAR(70)"), ("BillingCity", "NVARCHAR(40)"), ("BillingState", "NVARCHAR(40)"), ("BillingCountry", "NVARCHAR(40)"), ("BillingPostalCode", "NVARCHAR(10)"), ("Total", "NUMERIC(10,2)")],
        "InvoiceLine": [("InvoiceLineId", "INTEGER"), ("InvoiceId", "INTEGER"), ("TrackId", "INTEGER"), ("UnitPrice", "NUMERIC(10,2)"), ("Quantity", "INTEGER")],
        "MediaType": [("MediaTypeId", "INTEGER"), ("Name", "NVARCHAR(120)")],
        "Playlist": [("PlaylistId", "INTEGER"), ("Name", "NVARCHAR(120)")],
        "PlaylistTrack": [("PlaylistId", "INTEGER"), ("TrackId", "INTEGER")],
        "Track": [("TrackId", "INTEGER"), ("Name", "NVARCHAR(200)"), ("AlbumId", "INTEGER"), ("MediaTypeId", "INTEGER"), ("GenreId", "INTEGER"), ("Composer", "NVARCHAR(220)"), ("Milliseconds", "INTEGER"), ("Bytes", "INTEGER"), ("UnitPrice", "NUMERIC(10,2)")],
    },
    "pagila": {
        "customer": [("customer_id", "integer"), ("store_id", "integer"), ("first_name", "text"), ("last_name", "text"), ("email", "text"), ("address_id", "integer"), ("activebool", "boolean"), ("create_date", "date"), ("last_update", "timestamp with"), ("active", "integer")],
        "actor": [("actor_id", "integer"), ("first_name", "text"), ("last_name", "text"), ("last_update", "timestamp with")],
        "category": [("category_id", "integer"), ("name", "text"), ("last_update", "timestamp with")],
        "film": [("film_id", "integer"), ("title", "text"), ("description", "text"), ("release_year", "public.year"), ("language_id", "integer"), ("original_language_id", "integer"), ("rental_duration", "smallint"), ("rental_rate", "numeric(4,2)"), ("length", "smallint"), ("replacement_cost", "numeric(5,2)"), ("rating", "public.mpaa_rating"), ("last_update", "timestamp with"), ("special_features", "text"), ("fulltext", "tsvector")],
        "film_actor": [("actor_id", "integer"), ("film_id", "integer"), ("last_update", "timestamp with")],
        "film_category": [("film_id", "integer"), ("category_id", "integer"), ("last_update", "timestamp with")],
        "address": [("address_id", "integer"), ("address", "text"), ("address2", "text"), ("district", "text"), ("city_id", "integer"), ("postal_code", "text"), ("phone", "text"), ("last_update", "timestamp with")],
        "city": [("city_id", "integer"), ("city", "text"), ("country_id", "integer"), ("last_update", "timestamp with")],
        "country": [("country_id", "integer"), ("country", "text"), ("last_update", "timestamp with")],
        "inventory": [("inventory_id", "integer"), ("film_id", "integer"), ("store_id", "integer"), ("last_update", "timestamp with")],
        "language": [("language_id", "integer"), ("name", "character(20)"), ("last_update", "timestamp with")],
        "payment": [("payment_id", "integer"), ("customer_id", "integer"), ("staff_id", "integer"), ("rental_id", "integer"), ("amount", "numeric(5,2)"), ("payment_date", "timestamp with")],
        "rental": [("rental_id", "integer"), ("rental_date", "timestamp with"), ("inventory_id", "integer"), ("customer_id", "integer"), ("return_date", "timestamp with"), ("staff_id", "integer"), ("last_update", "timestamp with")],
        "staff": [("staff_id", "integer"), ("first_name", "text"), ("last_name", "text"), ("address_id", "integer"), ("email", "text"), ("store_id", "integer"), ("active", "boolean"), ("username", "text"), ("password", "text"), ("last_update", "timestamp with"), ("picture", "bytea")],
        "store": [("store_id", "integer"), ("manager_staff_id", "integer"), ("address_id", "integer"), ("last_update", "timestamp with")],
    },
}
```

## One text per table and per column

Each table and each column becomes one object whose text lists where it lives and what it is. The format follows the rows NVIDIA Auto Ontology embeds for its catalog (`auto_ontology/utils/embedding_rows.py`). The sample schemas have no column descriptions, so the text holds names and types only.

```python
def table_text(db, table, columns):
    cols = ",".join(f"{{name: {c}, data_type: {t}}}" for c, t in columns)
    return f"database_name: {db}, schema_name: public, table_name: {table}, columns: {cols}"

def column_text(db, table, column, data_type):
    return (
        f"database_name: {db}, schema_name: public, table_name: {table}, "
        f"column_name: {column}, data_type: {data_type}"
    )

rows = []
for db, tables in SCHEMA.items():
    for table, columns in tables.items():
        rows.append({"label": "Table", "key": f"{db}.{table}", "text": table_text(db, table, columns)})
        for column, data_type in columns:
            rows.append(
                {"label": "Column", "key": f"{db}.{table}.{column}", "text": column_text(db, table, column, data_type)}
            )

print(len(rows), "objects,", sum(r["label"] == "Column" for r in rows), "of them columns")
print(rows[1]["text"])
```

Python output:
```text
177 objects, 151 of them columns
database_name: chinook, schema_name: public, table_name: Album, column_name: AlbumId, data_type: INTEGER
```
## Embed the texts

We bring our own vectors from [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5). For this model, questions get a short instruction prefix and stored texts do not.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

doc_vectors = model.encode([r["text"] for r in rows], normalize_embeddings=True)
doc_vectors.shape
```

Python output:
```text
(177, 384)
```
## Create the collection and import

The collection has no vectorizer because we supply the vectors. A `flat` index searches exactly, which is fine for a few hundred objects. `label` and `key` use `field` tokenization so that filters match them whole.

```python
from weaviate.classes.config import Configure, DataType, Property, Tokenization, VectorDistances

if client.collections.exists("SchemaElement"):
    client.collections.delete("SchemaElement")

schema_elements = client.collections.create(
    "SchemaElement",
    vector_config=Configure.Vectors.self_provided(
        vector_index_config=Configure.VectorIndex.flat(distance_metric=VectorDistances.COSINE)
    ),
    properties=[
        Property(name="text", data_type=DataType.TEXT),
        Property(name="key", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
        Property(name="label", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    ],
)

with schema_elements.batch.fixed_size(batch_size=100) as batch:
    for row, vector in zip(rows, doc_vectors):
        batch.add_object(properties=row, vector=vector.tolist())

print("failed:", len(schema_elements.batch.failed_objects), "| stored:", len(schema_elements))
```

Python output:
```text
failed: 0 | stored: 177
```
## Questions with known answers

Twenty questions **name the column**, the way a coding agent that has seen the schema often asks. Twenty ask the same things in **plain English**, the way an analyst asks. Each lists the column (or columns) that answer it.

```python
NAMED = [
    ('rental_rate for each film', ['pagila.film.rental_rate']),
    ('average replacement_cost by rating', ['pagila.film.replacement_cost']),
    ('count rentals where return_date is null', ['pagila.rental.return_date']),
    ('rental_duration of films', ['pagila.film.rental_duration']),
    ('group payments by staff_id', ['pagila.payment.staff_id']),
    ('list customers with activebool false', ['pagila.customer.activebool']),
    ('manager_staff_id of each store', ['pagila.store.manager_staff_id']),
    ('films whose special_features include Trailers', ['pagila.film.special_features']),
    ('films grouped by rating', ['pagila.film.rating']),
    ('sum of Milliseconds per album', ['chinook.Track.Milliseconds']),
    ('SupportRepId for each customer', ['chinook.Customer.SupportRepId']),
    ('invoices by BillingCountry', ['chinook.Invoice.BillingCountry']),
    ('employees and their ReportsTo value', ['chinook.Employee.ReportsTo']),
    ('Track Bytes per track', ['chinook.Track.Bytes']),
    ('tracks with a Composer set', ['chinook.Track.Composer']),
    ('InvoiceLine Quantity totals', ['chinook.InvoiceLine.Quantity']),
    ('Employee HireDate', ['chinook.Employee.HireDate']),
    ('tracks by MediaTypeId', ['chinook.MediaType.Name', 'chinook.Track.MediaTypeId']),
    ('Invoice Total by month', ['chinook.Invoice.Total']),
    ('staff username list', ['pagila.staff.username']),
]

PLAIN = [
    ('How much does it cost to rent a movie?', ['pagila.film.rental_rate']),
    ('What do we charge a customer who loses a DVD?', ['pagila.film.replacement_cost']),
    ('Which rented movies have not been brought back yet?', ['pagila.rental.return_date']),
    ('How many days can a customer keep a rented movie?', ['pagila.film.rental_duration']),
    ('Which shop employee handled each payment?', ['pagila.payment.staff_id', 'pagila.rental.staff_id']),
    ("Is the customer's account still enabled?", ['pagila.customer.active', 'pagila.customer.activebool']),
    ('Who runs each store location?', ['pagila.store.manager_staff_id']),
    ('Which movies come with deleted scenes or behind the scenes extras?', ['pagila.film.special_features']),
    ('What age rating does each movie have, like PG or R?', ['pagila.film.rating']),
    ('How long is each song?', ['chinook.Track.Milliseconds']),
    ('Which sales agent looks after each client?', ['chinook.Customer.SupportRepId']),
    ('Which country were the bills sent to?', ['chinook.Invoice.BillingCountry']),
    ("Who is each staff member's boss?", ['chinook.Employee.ReportsTo']),
    ('How big is each audio file on disk?', ['chinook.Track.Bytes']),
    ('Who wrote the music for each song?', ['chinook.Track.Composer']),
    ('How many copies of a song were bought on one order line?', ['chinook.InvoiceLine.Quantity']),
    ('When did each employee join the company?', ['chinook.Employee.HireDate']),
    ('What file format is each song in, like MPEG or AAC?', ['chinook.MediaType.Name', 'chinook.Track.MediaTypeId']),
    ('How much did each order cost in total?', ['chinook.Invoice.Total']),
    ('What is the login name of each rental store employee?', ['pagila.staff.username']),
]
```

## Three ways to search

All three search columns only and return the top 10, which is the kind of shortlist a text-to-SQL agent passes on to its LLM.

- **Vector**: nearest vectors by cosine distance.
- **Keyword**: BM25 on the `text` property.
- **Hybrid**: both, fused. `alpha=1` is pure vector and `alpha=0` is pure keyword.

```python
from weaviate.classes.query import Filter

ONLY_COLUMNS = Filter.by_property("label").equal("Column")
K = 10

def keys(response):
    return [o.properties["key"] for o in response.objects]

def vector_search(question, qvec):
    return keys(schema_elements.query.near_vector(near_vector=qvec, limit=K, filters=ONLY_COLUMNS))

def keyword_search(question, qvec):
    return keys(
        schema_elements.query.bm25(query=question, query_properties=["text"], limit=K, filters=ONLY_COLUMNS)
    )

def hybrid_search(alpha):
    def search(question, qvec):
        return keys(
            schema_elements.query.hybrid(
                query=question, vector=qvec, alpha=alpha, query_properties=["text"], limit=K, filters=ONLY_COLUMNS
            )
        )

    return search
```

## Measure

For each question we find the rank of the first correct column. **hit@k** is the share of questions with a correct column in the top k. **MRR** (mean reciprocal rank) averages 1/rank and rewards putting the answer first.

```python
import pandas as pd

QUESTION_SETS = {"names the column": NAMED, "plain English": PLAIN}
QVECS = {
    name: model.encode([QUERY_PREFIX + q for q, _ in qs], normalize_embeddings=True)
    for name, qs in QUESTION_SETS.items()
}

def evaluate(search, set_name):
    hits, rr = {1: 0, 5: 0, 10: 0}, 0.0
    questions = QUESTION_SETS[set_name]
    for (question, gold), qvec in zip(questions, QVECS[set_name]):
        ranked = search(question, qvec.tolist())
        rank = next((i + 1 for i, key in enumerate(ranked) if key in gold), None)
        if rank:
            rr += 1 / rank
            for k in hits:
                hits[k] += k >= rank
    n = len(questions)
    return {"hit@1": hits[1] / n, "hit@5": hits[5] / n, "hit@10": hits[10] / n, "MRR": rr / n}

METHODS = {
    "vector": vector_search,
    "hybrid alpha=0.75": hybrid_search(0.75),
    "hybrid alpha=0.5": hybrid_search(0.5),
    "hybrid alpha=0.25": hybrid_search(0.25),
    "keyword (BM25)": keyword_search,
}

results = pd.DataFrame(
    [{"questions": s, "search": m, **evaluate(fn, s)} for s in QUESTION_SETS for m, fn in METHODS.items()]
)
results.round(2)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type \{
        vertical-align: middle;
    }

    .dataframe tbody tr th \{
        vertical-align: top;
    }

    .dataframe thead th \{
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr  style={{textAlign: 'right'}}>
      <th></th>
      <th>questions</th>
      <th>search</th>
      <th>hit@1</th>
      <th>hit@5</th>
      <th>hit@10</th>
      <th>MRR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>names the column</td>
      <td>vector</td>
      <td>0.90</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>1</th>
      <td>names the column</td>
      <td>hybrid alpha=0.75</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>names the column</td>
      <td>hybrid alpha=0.5</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>names the column</td>
      <td>hybrid alpha=0.25</td>
      <td>0.95</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.98</td>
    </tr>
    <tr>
      <th>4</th>
      <td>names the column</td>
      <td>keyword (BM25)</td>
      <td>0.95</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>5</th>
      <td>plain English</td>
      <td>vector</td>
      <td>0.35</td>
      <td>0.60</td>
      <td>0.75</td>
      <td>0.46</td>
    </tr>
    <tr>
      <th>6</th>
      <td>plain English</td>
      <td>hybrid alpha=0.75</td>
      <td>0.25</td>
      <td>0.60</td>
      <td>0.75</td>
      <td>0.41</td>
    </tr>
    <tr>
      <th>7</th>
      <td>plain English</td>
      <td>hybrid alpha=0.5</td>
      <td>0.10</td>
      <td>0.60</td>
      <td>0.75</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>8</th>
      <td>plain English</td>
      <td>hybrid alpha=0.25</td>
      <td>0.10</td>
      <td>0.60</td>
      <td>0.75</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>9</th>
      <td>plain English</td>
      <td>keyword (BM25)</td>
      <td>0.10</td>
      <td>0.25</td>
      <td>0.30</td>
      <td>0.15</td>
    </tr>
  </tbody>
</table>
</div>

### What the numbers say

- **Questions that name the column:** every search finds the right column in the top 10. Hybrid search with `alpha` 0.5 or 0.75 also puts it **first** every time, which vector search alone does for 90% of them.
- **Plain-English questions:** vector search does best. Giving keywords more weight makes the first result worse (hit@1 drops from 0.35 to 0.10), because words like "cost" or "boss" never appear in the column texts. With vector or hybrid search, a quarter of these questions still miss the top 10.

These are the plain-English questions where vector search's first result was wrong:

```python
for (question, gold), qvec in zip(PLAIN, QVECS["plain English"]):
    top3 = vector_search(question, qvec.tolist())[:3]
    if top3[0] not in gold:
        print(f"{question}\n  expected: {gold}\n  got:      {top3}\n")
```

Python output:
```text
Which rented movies have not been brought back yet?
  expected: ['pagila.rental.return_date']
  got:      ['pagila.film.rental_duration', 'pagila.film.rental_rate', 'pagila.payment.rental_id']

Which shop employee handled each payment?
  expected: ['pagila.payment.staff_id', 'pagila.rental.staff_id']
  got:      ['pagila.payment.payment_id', 'pagila.payment.customer_id', 'pagila.payment.staff_id']

Who runs each store location?
  expected: ['pagila.store.manager_staff_id']
  got:      ['pagila.inventory.store_id', 'pagila.customer.store_id', 'pagila.store.store_id']

Which movies come with deleted scenes or behind the scenes extras?
  expected: ['pagila.film.special_features']
  got:      ['pagila.film_actor.film_id', 'pagila.film_category.film_id', 'pagila.film.special_features']

How long is each song?
  expected: ['chinook.Track.Milliseconds']
  got:      ['chinook.Playlist.PlaylistId', 'chinook.PlaylistTrack.PlaylistId', 'chinook.PlaylistTrack.TrackId']

Which sales agent looks after each client?
  expected: ['chinook.Customer.SupportRepId']
  got:      ['pagila.customer.activebool', 'chinook.Customer.SupportRepId', 'chinook.Invoice.CustomerId']

Which country were the bills sent to?
  expected: ['chinook.Invoice.BillingCountry']
  got:      ['chinook.Invoice.BillingPostalCode', 'chinook.Invoice.BillingCountry', 'chinook.Invoice.BillingCity']

Who is each staff member's boss?
  expected: ['chinook.Employee.ReportsTo']
  got:      ['pagila.staff.staff_id', 'pagila.store.manager_staff_id', 'pagila.staff.email']

How big is each audio file on disk?
  expected: ['chinook.Track.Bytes']
  got:      ['pagila.film.length', 'pagila.film.rental_duration', 'pagila.film.rating']

Who wrote the music for each song?
  expected: ['chinook.Track.Composer']
  got:      ['chinook.Playlist.PlaylistId', 'chinook.PlaylistTrack.PlaylistId', 'chinook.Album.ArtistId']

How many copies of a song were bought on one order line?
  expected: ['chinook.InvoiceLine.Quantity']
  got:      ['chinook.Playlist.PlaylistId', 'chinook.PlaylistTrack.PlaylistId', 'chinook.Album.ArtistId']

What file format is each song in, like MPEG or AAC?
  expected: ['chinook.MediaType.Name', 'chinook.Track.MediaTypeId']
  got:      ['pagila.film.rating', 'pagila.film_category.film_id', 'pagila.film.special_features']

What is the login name of each rental store employee?
  expected: ['pagila.staff.username']
  got:      ['chinook.Employee.HireDate', 'pagila.rental.inventory_id', 'pagila.payment.rental_id']
```
## A simple router

Since neither setting wins both kinds of question, pick per question: if any word in the question is an actual column name, use hybrid search, otherwise use vector search.

```python
import re

COLUMN_NAMES = {r["key"].split(".")[-1].lower() for r in rows if r["label"] == "Column"}

def names_a_column(question):
    return any(word.lower() in COLUMN_NAMES for word in re.findall(r"\w+", question))

def routed_search(question, qvec):
    if names_a_column(question):
        return hybrid_search(0.5)(question, qvec)
    return vector_search(question, qvec)

print("sent to hybrid:", {s: f"{sum(names_a_column(q) for q, _ in qs)}/{len(qs)}" for s, qs in QUESTION_SETS.items()})
print("plain-English ones:", [q for q, _ in PLAIN if names_a_column(q)])

routed = pd.DataFrame(
    [{"questions": s, "search": m, **evaluate(fn, s)}
     for s in QUESTION_SETS
     for m, fn in {"vector": vector_search, "hybrid alpha=0.5": hybrid_search(0.5), "router": routed_search}.items()]
)
routed.round(2)
```

Python output:
```text
sent to hybrid: {'names the column': '20/20', 'plain English': '5/20'}
plain-English ones: ['What age rating does each movie have, like PG or R?', 'Which country were the bills sent to?', 'When did each employee join the company?', 'How much did each order cost in total?', 'What is the login name of each rental store employee?']
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type \{
        vertical-align: middle;
    }

    .dataframe tbody tr th \{
        vertical-align: top;
    }

    .dataframe thead th \{
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr  style={{textAlign: 'right'}}>
      <th></th>
      <th>questions</th>
      <th>search</th>
      <th>hit@1</th>
      <th>hit@5</th>
      <th>hit@10</th>
      <th>MRR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>names the column</td>
      <td>vector</td>
      <td>0.90</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>1</th>
      <td>names the column</td>
      <td>hybrid alpha=0.5</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>names the column</td>
      <td>router</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>plain English</td>
      <td>vector</td>
      <td>0.35</td>
      <td>0.6</td>
      <td>0.75</td>
      <td>0.46</td>
    </tr>
    <tr>
      <th>4</th>
      <td>plain English</td>
      <td>hybrid alpha=0.5</td>
      <td>0.10</td>
      <td>0.6</td>
      <td>0.75</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>5</th>
      <td>plain English</td>
      <td>router</td>
      <td>0.30</td>
      <td>0.6</td>
      <td>0.75</td>
      <td>0.43</td>
    </tr>
  </tbody>
</table>
</div>

The router sends every named-column question to hybrid search and keeps its perfect first-place score. It also sends five plain-English questions to hybrid, because ordinary words such as `rating`, `country` and `total` happen to be column names. That costs one first-place hit (0.30 against 0.35 for vector alone).

## Takeaways

- No single `alpha` is best for both kinds of question. Measure on your own questions before choosing one.
- When questions use the schema's own words, hybrid search ranks the right column first more often.
- When questions use business words that never appear in the schema, keyword matching adds noise. Most remaining misses need better stored text (descriptions, synonyms), not a different search setting.
- A cheap router based on the schema's real column names recovers most of the benefit of both.

**Caveats.** This is a small test: 151 columns, 40 questions written for this recipe, one small embedding model, and no column descriptions or sample values. Larger schemas with many similar column names may behave differently.

## Next steps

- Add column descriptions, or business terms with synonyms, to the stored text. These carry the words plain-English questions use.
- Store a `certified` boolean on each definition and add `Filter.by_property("certified").equal(True)` so that the agent only links human-reviewed definitions.
- Re-rank the shortlist with a [reranker](https://docs.weaviate.io/weaviate/search/rerank).

## Clean up

```python
client.collections.delete("SchemaElement")
client.close()
```