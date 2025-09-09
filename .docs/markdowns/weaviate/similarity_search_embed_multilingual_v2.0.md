---
layout: recipe
toc: True
title: "Similarity Search with Cohere"
featured: False
integration: False
agent: False
tags: ['Similarity Search', 'Cohere']
---
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weaviate/recipes/blob/main/weaviate-features/model-providers/cohere/similarity_search_embed_multilingual_v2.0.ipynb)

## Dependencies

```python
!pip install weaviate-client
```

## Connect to Weaviate

```python
import weaviate, os

# Connect to your local Weaviate instance deployed with Docker
client = weaviate.connect_to_local(
  headers={
    "X-COHERE-Api-Key": os.environ["COHERE_API_KEY"] # Replace with your Cohere key - recommended: use env var
  }
)

# Option 2
# Connect to your Weaviate Client Service cluster
# client = weaviate.connect_to_wcs(
#     cluster_url="WCS-CLUSTER-ID",                             # Replace with your WCS cluster ID
#     auth_credentials=weaviate.auth.AuthApiKey("WCS-API-KEY"), # Replace with your WCS API KEY - recommended: use env var
#     headers={
#         "X-Cohere-Api-Key": os.getenv("COHERE_API_KEY"),      # Replace with your inference API key - recommended: use env var
#     }
# )

client.is_ready()
```

## Create a collection
> Collection stores your data and vector embeddings.

```python
# Note: in practice, you shouldn"t rerun this cell, as it deletes your data
# in "JeopardyQuestion", and then you need to re-import it again.
import weaviate.classes.config as wc

# Delete the collection if it already exists
if (client.collections.exists("JeopardyQuestion")):
    client.collections.delete("JeopardyQuestion")

client.collections.create(
    name="JeopardyQuestion",

    vector_config=wc.Configure.Vectors.text2vec_cohere( # specify the vectorizer and model type you"re using
        model="embed-multilingual-v2.0",                       # defaults to embed-multilingual-v2.0 if not set
    ),

    properties=[ # defining properties (data schema) is optional
        wc.Property(name="Question", data_type=wc.DataType.TEXT), 
        wc.Property(name="Answer", data_type=wc.DataType.TEXT),
        wc.Property(name="Category", data_type=wc.DataType.TEXT, skip_vectorization=True), 
    ]
)

print("Successfully created collection: JeopardyQuestion.")
```

## Import the Data

```python
import requests, json
url = "https://raw.githubusercontent.com/weaviate/weaviate-examples/main/jeopardy_small_dataset/jeopardy_tiny.json"
resp = requests.get(url)
data = json.loads(resp.text)

# Get a collection object for "JeopardyQuestion"
jeopardy = client.collections.use("JeopardyQuestion")

# Insert data objects
response = jeopardy.data.insert_many(data)

# Note, the `data` array contains 10 objects, which is great to call insert_many with.
# However, if you have a milion objects to insert, then you should spit them into smaller batches (i.e. 100-1000 per insert)

if (response.has_errors):
    print(response.errors)
else:
    print("Insert complete.")
```

## Query Weaviate: Similarity Search (Text objects)

Similarity search options for text objects in **Weaviate**:

1. [near_text](https://docs.weaviate.io/weaviate/search/similarity#an-input-medium)

2. [near_object](https://docs.weaviate.io/weaviate/search/similarity#an-object)

3. [near_vector](https://docs.weaviate.io/weaviate/search/similarity#a-vector)

### nearText Example

Find a `JeopardyQuestion` about "animals in movies". Limit it to only 4 responses.

```python
# note, you can reuse the collection object from the previous cell.
# Get a collection object for "JeopardyQuestion"
jeopardy = client.collections.use("JeopardyQuestion")

response = jeopardy.query.near_text(
    query="african beasts",
    limit=4
)

for item in response.objects:
    print("ID:", item.uuid)
    print("Data:", json.dumps(item.properties, indent=2), "\n")
```

Return vector embeddings.

```python
response = jeopardy.query.near_text(
    query="african beasts",
    include_vector=True,
    limit=4
)

for item in response.objects:
    print("ID:", item.uuid)
    print("Data:", json.dumps(item.properties, indent=2))
    print("Vector:", item.vector, "\n")
```

Now, also request the `distance` for each returned item.

```python
import weaviate.classes.query as wq

response = jeopardy.query.near_text(
    query="african beasts",
    return_metadata=wq.MetadataQuery(distance=True),
    limit=4
)

for item in response.objects:
    print("ID:", item.uuid)
    print("Distance:", item.metadata.distance)
    print("Data:", item.properties, "\n")
```

### nearObject Example

Search through the `JeopardyQuestion` class to find the top 4 objects closest to id `a1dd67f9-bfa7-45e1-b45e-26eb8c52e9a6`. (The id was taken from the query above)

```python
response = jeopardy.query.near_object(
    near_object="a1dd67f9-bfa7-45e1-b45e-26eb8c52e9a6", # replace with your id of interest
    limit=4
)

for item in response.objects:
    print("ID:", item.uuid)
    print("Data:", item.properties, "\n")
```

### nearVector Example

Search through the `JeopardyQuestion` class to find the top 2 objects closest to the query vector `[-0.0125526935, -0.021168863, ... ]`

```python
response = jeopardy.query.near_vector(
    near_vector=[-0.0125526935, -0.021168863, ... ], # your vector object goes here
    limit=4
)

for item in response.objects:
    print("ID:", item.uuid)
    print("Data:", item.properties, "\n")
```