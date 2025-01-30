---
layout: recipe
colab: https://colab.research.google.com/github/weaviate/recipes/blob/main//weaviate-services/weaviate_embeddings_service.ipynb
toc: True
title: "How to Use Weaviate Embedding Service"
download: /downloads/weaviate_embeddings_service.ipynb
featured: True
tags: ['Weaviate Embeddings', 'Weaviate Cloud']
---
    


[Weaviate Embeddings](https://weaviate.io/developers/wcs/embeddings) enables you to generate embeddings directly from a [Weaviate Cloud](https://console.weaviate.cloud/) database instance. 

*Please note this service is part of Weaviate Cloud and cannot be accessed through open-source. Additionally, this service is currently under technical preview, and you can request access [here](https://events.weaviate.io/embeddings-preview).*

## Import Dependencies, Libraries, and Keys


```python
!pip install --q weaviate-client
```


```python
!pip show weaviate-client # you need to have the Python client version 4.9.5 or higher
```


```python
import weaviate
from weaviate.classes.init import Auth
import os
import weaviate.classes.config as wc

import requests, json
```


```python
WCD_CLUSTER_URL = os.getenv("WCD_CLUSTER_URL")
WCD_CLUSTER_KEY = os.getenv("WCD_CLUSTER_KEY")
```

## Connect to Weaviate


```python
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WCD_CLUSTER_URL,
    auth_credentials=Auth.api_key(WCD_CLUSTER_KEY),
)

print(client.is_ready())
```

    True


## Define Collection


```python
# Note: This will delete your data stored in "JeopardyQuestion".and
# It will require you to re-import again.

# Delete the collection if it already exists
if (client.collections.exists("JeopardyQuestion")):
    client.collections.delete("JeopardyQuestion")

client.collections.create(
    name="JeopardyQuestion",

    vectorizer_config=wc.Configure.Vectorizer.text2vec_weaviate( # specify the vectorizer and model type you're using
        model="arctic-embed-m-v1.5", # default model
    ),

    properties=[ # defining properties (data schema) is optional
        wc.Property(name="Question", data_type=wc.DataType.TEXT), 
        wc.Property(name="Answer", data_type=wc.DataType.TEXT),
        wc.Property(name="Category", data_type=wc.DataType.TEXT) 
    ]
)

print("Successfully created collection: JeopardyQuestion.")
```

    Successfully created collection: JeopardyQuestion.


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

    Insert complete.



```python
# count the number of objects

jeopardy = client.collections.get("JeopardyQuestion")
response = jeopardy.aggregate.over_all(total_count=True)

print(response.total_count)
```

    10


## Hybrid Search

The goal of this notebook is to show you how to use the embedding service. For more information on hybrid search, check out [this folder](/weaviate-features/hybrid-search/) and/or the [documentation](https://weaviate.io/developers/weaviate/search/hybrid).


```python
jeopardy = client.collections.get("JeopardyQuestion")

response = jeopardy.query.hybrid(
    query="northern beast",
    alpha=0.8,
    limit=3
)

for item in response.objects:
    print("ID:", item.uuid)
    print("Data:", json.dumps(item.properties, indent=2), "\n")
```

    ID: 3d34d4a2-30cb-4268-988a-797a322520d9
    Data: {
      "answer": "species",
      "question": "2000 news: the Gunnison sage grouse isn't just another northern sage grouse, but a new one of this classification",
      "category": "SCIENCE"
    } 
    
    ID: 3b0803e4-8c5b-4d17-a216-46ec65e07dd4
    Data: {
      "answer": "Antelope",
      "question": "Weighing around a ton, the eland is the largest species of this animal in Africa",
      "category": "ANIMALS"
    } 
    
    ID: b569b667-d0c2-4ec7-82e6-9634d721b231
    Data: {
      "answer": "Elephant",
      "question": "It's the only living mammal in the order Proboseidea",
      "category": "ANIMALS"
    } 
    

