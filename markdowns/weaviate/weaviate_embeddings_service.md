---
layout: recipe
colab: https://colab.research.google.com/github/weaviate/recipes/blob/main/weaviate-services/embedding-service/weaviate_embeddings_service.ipynb
toc: True
title: "How to Use Weaviate Embedding Service"
featured: True
integration: False
agent: False
tags: ['Weaviate Embeddings', 'Weaviate Cloud']
---
# Weaviate Embedding Service

[Weaviate Embeddings](https://weaviate.io/developers/wcs/embeddings) enables you to generate embeddings directly from a [Weaviate Cloud](https://console.weaviate.cloud/) database instance. 

*Please note this service is part of Weaviate Cloud and cannot be accessed through open-source. Additionally, this service is currently under technical preview, and you can request access [here](https://events.weaviate.io/embeddings-preview).*

This notebook will show you how to:
1. Define a Weaviate Collection
1. Run a vector search query 
1. Run a hybrid search query
1. Run a hybrid search query with metadata filters
1. Run a generative search query (RAG)

## Requirements

1. Weaviate Cloud (WCD) account: You can register [here](https://console.weaviate.cloud/)
1. Create a cluster on WCD: A sandbox or serverless cluster is fine. You will need to grab the cluster URL and admin API key
1. OpenAI key to access `GPT-4o mini`

```python
!pip install --q weaviate-client
```

```python
!pip show weaviate-client # you need to have the Python client version 4.9.5 or higher
```

## Import Libraries and Keys

```python
import weaviate
from weaviate.classes.init import Auth
import os
import weaviate.classes.config as wc
from weaviate.classes.query import Filter

import requests, json
import pandas as pd
from io import StringIO
```

```python
WCD_CLUSTER_URL = os.getenv("WCD_CLUSTER_URL")
WCD_CLUSTER_KEY = os.getenv("WCD_CLUSTER_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

## Connect to Weaviate

```python
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WCD_CLUSTER_URL,
    auth_credentials=Auth.api_key(WCD_CLUSTER_KEY),

    headers={
        "X-OpenAI-Api-Key": OPENAI_API_KEY,
    }
)

print(client.is_ready())
```

Python output:
```text
True
```
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
        model="Snowflake/snowflake-arctic-embed-l-v2.0", # default model
    ),

    generative_config=wc.Configure.Generative.openai( 
        model="gpt-4o-mini" # select model, default is gpt-3.5-turbo 
    ),

    properties=[ # defining properties (data schema) is optional
        wc.Property(name="Question", data_type=wc.DataType.TEXT), 
        wc.Property(name="Answer", data_type=wc.DataType.TEXT, skip_vectorization=True),
        wc.Property(name="Category", data_type=wc.DataType.TEXT, skip_vectorization=True),
        wc.Property(name="Value", data_type=wc.DataType.TEXT, skip_vectorization=True)
    ]
)

print("Successfully created collection: JeopardyQuestion.")
```

Python output:
```text
Successfully created collection: JeopardyQuestion.
```
## Import Data

We will use the small jeopardy dataset as an example. It has 1,000 objects.

```python
url = 'https://raw.githubusercontent.com/weaviate/weaviate-examples/main/jeopardy_small_dataset/jeopardy_small.csv'
resp = requests.get(url)

df = pd.read_csv(StringIO(resp.text))
```

```python
# Get a collection object for "JeopardyQuestion"
collection = client.collections.get("JeopardyQuestion")

# Insert data objects with batch import
with collection.batch.dynamic() as batch:
    for _, row in df.iterrows():
        properties = {
            "question": row['Question'],
            "answer": row['Answer'],
            "category": row["Category"],
            "value": row["Value"]
        }
        batch.add_object(properties)

failed_objects = collection.batch.failed_objects
if failed_objects:
    print(f"Number of failed imports: {len(failed_objects)}")
else:
    print("Insert complete.")
```

Python output:
```text
Insert complete.
```
```python
# count the number of objects

collection = client.collections.get("JeopardyQuestion")
response = collection.aggregate.over_all(total_count=True)

print(response.total_count)
```

Python output:
```text
1000
```
## Query Time

### Vector Search

```python
collection = client.collections.get("JeopardyQuestion")

response = collection.query.near_text(
    query="marine mamal with tusk", 
    limit=2 # limit to only 2
)

for item in response.objects:
    print("Data:", json.dumps(item.properties, indent=2), "\n")
```

Python output:
```text
Data: {
  "value": "NaN",
  "answer": "the narwhal",
  "question": "A part of this marine mammal was prized by medieval folk, who thought it belonged to a unicorn",
  "category": "THE ANIMAL KINGDOM"
} 

Data: {
  "value": "$400",
  "answer": "the walrus",
  "question": "You could say this Arctic mammal, Odobenus rosmarus, has a Wilford Brimley mustache",
  "category": "MAMMALS"
} 
```
### Hybrid Search

The goal of this notebook is to show you how to use the embedding service. For more information on hybrid search, check out [this folder](/weaviate-features/hybrid-search/) and/or the [documentation](https://weaviate.io/developers/weaviate/search/hybrid).

The `alpha` parameter determines the weight given to the sparse and dense search methods. `alpha = 0` is pure sparse (bm25) search, whereas `alpha = 1` is pure dense (vector) search. 

Alpha is an optional parameter. The default is set to `0.75`.

```python
collection = client.collections.get("JeopardyQuestion")

response = collection.query.hybrid(
    query="unicorn-like artic animal",
    alpha=0.7, 
    limit=2
)

for item in response.objects:
    print("Data:", json.dumps(item.properties, indent=2), "\n")
```

Python output:
```text
Data: {
  "value": "NaN",
  "answer": "the narwhal",
  "question": "A part of this marine mammal was prized by medieval folk, who thought it belonged to a unicorn",
  "category": "THE ANIMAL KINGDOM"
} 

Data: {
  "value": "$400",
  "answer": "the walrus",
  "question": "You could say this Arctic mammal, Odobenus rosmarus, has a Wilford Brimley mustache",
  "category": "MAMMALS"
} 
```
### Fetch Objects with Metadata Filters

Learn more about the different filter operators [here](https://weaviate.io/developers/weaviate/search/filters).

```python
collection = client.collections.get("JeopardyQuestion")

response = collection.query.fetch_objects(
    limit=2,
    filters=Filter.by_property("category").equal("BUSINESS & INDUSTRY")
)

for item in response.objects:
    print("Data:", json.dumps(item.properties, indent=2), "\n")
```

Python output:
```text
Data: {
  "value": "$200",
  "answer": "Disney",
  "question": "This company operates the 4 most popular theme parks in North America",
  "category": "BUSINESS & INDUSTRY"
} 

Data: {
  "value": "$400",
  "answer": "Yamaha",
  "question": "This firm began in 1897 as Nippon Gakki Company, an organ manufacturer; electronic organs came along in 1959",
  "category": "BUSINESS & INDUSTRY"
} 
```
### Generative Search (RAG)

```python
collection = client.collections.get("JeopardyQuestion")

response = collection.generate.hybrid(
    query="unicorn-like artic animal",
    alpha=0.7, 
    grouped_task="Explain why people thought these animals were unicorn-like",
    limit=2
)

print(f"Generated output: {response.generated}") 
```

Python output:
```text
Generated output: People thought these animals were unicorn-like for a few reasons:

1. **Narwhal**: The narwhal is a marine mammal known for its long, spiral tusk, which can reach lengths of up to 10 feet. In medieval times, this tusk was often sold as a "unicorn horn" and was believed to possess magical properties. The resemblance of the narwhal's tusk to the mythical unicorn's horn led to the association between the two, as people were fascinated by the idea of unicorns and sought to find evidence of their existence in the natural world.

2. **Walrus**: While the walrus does not have a direct connection to unicorns like the narwhal, its large tusks and unique appearance may have contributed to some fantastical interpretations. The walrus's tusks, which can be quite prominent, might have sparked the imagination of those who were already inclined to believe in mythical creatures. Additionally, the walrus's size and distinctive features could have led to comparisons with other legendary animals, including unicorns, in folklore and storytelling.

Overall, the combination of physical characteristics and the cultural context of the time contributed to the perception of these animals as unicorn-like.
```