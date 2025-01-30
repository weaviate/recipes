---
layout: recipe
colab: https://colab.research.google.com/github/weaviate/recipes/blob/main//weaviate-features/hybrid-search/hybrid_search_mistral.ipynb
toc: True
title: "Hybrid Search with Mistral"
download: /downloads/hybrid_search_mistral.ipynb
featured: False
tags: ['Hybrid Search', 'Mistral']
---
    


This recipe will show you how to run hybrid search with embeddings from Mistral.

## Requirements

1. Weaviate cluster
    1. You can create a 14-day free sandbox on [WCD](https://console.weaviate.cloud/)
    2. [Embedded Weaviate](https://weaviate.io/developers/weaviate/installation/embedded)
    3. [Local deployment](https://weaviate.io/developers/weaviate/installation/docker-compose#starter-docker-compose-file)
    4. [Other options](https://weaviate.io/developers/weaviate/installation)

2. Mistral API key. Grab one [here](https://docs.mistral.ai/api/).


```python
import weaviate, os
from weaviate.embedded import EmbeddedOptions
import weaviate.classes as wvc
import weaviate.classes.config as wc
import requests, json
import weaviate.classes.query as wq
```

## Connect to Weaviate

Only choose one option from the below.

**Weaviate Cloud Deployment**


```python
WCD_URL = os.environ["WEAVIATE_URL"] # Replace with your Weaviate cluster URL
WCD_AUTH_KEY = os.environ["WEAVIATE_AUTH"] # Replace with your cluster auth key
MISTRAL_KEY = os.environ["MISTRAL_API_KEY"] # Replace with your Mistral key

# Weaviate Cloud Deployment
client = weaviate.connect_to_wcs(
    cluster_url=WCD_URL,
    auth_credentials=weaviate.auth.AuthApiKey(WCD_AUTH_KEY),
      headers={ "X-Mistral-Api-Key": MISTRAL_KEY}
)

print(client.is_ready())
```

**Embedded Weaviate**


```python
# MISTRAL_KEY = os.environ["MISTRAL_API_KEY"] # Replace with your Mistral key

# client = weaviate.WeaviateClient(
#     embedded_options=EmbeddedOptions(
#         version="1.26.1",
#         additional_env_vars={
#             "ENABLE_MODULES": "text2vec-mistral"
#         }),
#         additional_headers={
#             "X-Mistral-Api-Key": MISTRAL_KEY
#         }
# )

# client.connect()
```

**Local Deployment**


```python
# MISTRAL_KEY = os.environ["MISTRAL_API_KEY"] # Replace with your Mistral key

# client = weaviate.connect_to_local(
#   headers={
#     "X-Mistral-Api-Key": MISTRAL_KEY
#   }
# )
# print(client.is_ready())
```

## Create a collection
> Collection stores your data and vector embeddings.


```python
# Note: in practice, you shouldn't rerun this cell, as it deletes your data
# in "JeopardyQuestion", and then you need to re-import it again.

# Delete the collection if it already exists
if (client.collections.exists("JeopardyQuestion")):
    client.collections.delete("JeopardyQuestion")

client.collections.create(
    name="JeopardyQuestion",

    vectorizer_config=wc.Configure.Vectorizer.text2vec_mistral( # specify the vectorizer and model
        model="mistral-embed",
    ),

    properties=[ # defining properties (data schema) is optional
        wc.Property(name="Question", data_type=wc.DataType.TEXT), 
        wc.Property(name="Answer", data_type=wc.DataType.TEXT),
        wc.Property(name="Category", data_type=wc.DataType.TEXT, skip_vectorization=True), 
    ]
)

print("Successfully created collection: JeopardyQuestion.")
```

## Import Data


```python
url = 'https://raw.githubusercontent.com/weaviate/weaviate-examples/main/jeopardy_small_dataset/jeopardy_tiny.json'
resp = requests.get(url)
data = json.loads(resp.text)

# Get a collection object for "JeopardyQuestion"
jeopardy = client.collections.get("JeopardyQuestion")

# Insert data objects
response = jeopardy.data.insert_many(data)

# Note, the `data` array contains 10 objects, which is great to call insert_many with.
# However, if you have a milion objects to insert, then you should spit them into smaller batches (i.e. 100-1000 per insert)

if (response.has_errors):
    print(response.errors)
else:
    print("Insert complete.")
```

## Hybrid Search

The `alpha` parameter determines the weight given to the sparse and dense search methods. `alpha = 0` is pure sparse (bm25) search, whereas `alpha = 1` is pure dense (vector) search. 

Alpha is an optional parameter. The default is set to `0.75`.

### Hybrid Search only

The below query is finding Jeopardy questions about animals and is limiting the output to only two results. Notice `alpha` is set to `0.80`, which means it is weighing the vector search results more than bm25. If you were to set `alpha = 0.25`, you would get different results. 


```python
response = jeopardy.query.hybrid(
    query="northern beast",
    query_properties=["question"],
    alpha=0.8,
    limit=3
)

for item in response.objects:
    print("ID:", item.uuid)
    print("Data:", json.dumps(item.properties, indent=2), "\n")
```

### Hybrid Search with a `where` filter

Find Jeopardy questions about elephants, where the category is set to Animals.


```python
response = jeopardy.query.hybrid(
    query="northern beast",
    alpha=0.8,
    filters=wq.Filter.by_property("category").equal("Animals"),
    limit=3
)

for item in response.objects:
    print("ID:", item.uuid)
    print("Data:", json.dumps(item.properties, indent=2), "\n")
```
