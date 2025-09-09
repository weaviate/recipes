---
layout: recipe
toc: True
title: "Generative search (RAG) with AWS Bedrock"
featured: False
integration: False
agent: False
tags: ['Generative Search', 'RAG', 'AWS']
---
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weaviate/recipes/weaviate-features/model-providers/aws/rag_titan-text-express-v1_bedrock.ipynb)

## Dependencies

```python
!pip install weaviate-client
```

## Configuration

```python
import weaviate, os

# Connect to your local Weaviate instance deployed with Docker
client = weaviate.connect_to_local(
    headers={
        "X-AWS-Access-Key": os.getenv("AWS_ACCESS_KEY"), # Replace with your AWS access key - recommended: use env var
        "X-AWS-Secret-Key": os.getenv("AWS_SECRET_KEY"), # Replace with your AWS secret key - recommended: use env var
    }
)

# Option 2
# Connect to your Weaviate Client Service cluster
# client = weaviate.connect_to_wcs(
#     cluster_url="WCS-CLUSTER-ID",                             # Replace with your WCS cluster ID
#     auth_credentials=weaviate.auth.AuthApiKey("WCS-API-KEY"), # Replace with your WCS API KEY - recommended: use env var
#     headers={
#         "X-AWS-Access-Key": os.getenv("AWS_ACCESS_KEY"), # Replace with your AWS access key - recommended: use env var
#         "X-AWS-Secret-Key": os.getenv("AWS_SECRET_KEY"), # Replace with your AWS secret key - recommended: use env var
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

    vectorizer_config=wc.Configure.Vectorizer.text2vec_aws(
        service="bedrock",   #this is crucial
        model="cohere.embed-english-v3", # select the model, make sure it is enabled for your account
        # model="amazon.titan-embed-text-v1", # select the model, make sure it is enabled for your account
        region="eu-west-2"               # select your region
    ),

    # Enable generative model from AWS
    generative_config=wc.Configure.Generative.aws(
        service="bedrock",   #this is crucial
        model="amazon.titan-text-express-v1", # select the model, make sure it is enabled for your account
        region="eu-west-2"               # select your region
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

## Generative Search Queries

### Single Result

Single Result makes a generation for each individual search result. 

In the below example, I want to create a Facebook ad from the Jeopardy question about Elephants. 

```python
generatePrompt = "Turn the following Jeogrady question into a Facebook Ad: {question}"

jeopardy = client.collections.get("JeopardyQuestion")
response = jeopardy.generate.near_text(
    query="Elephants",
    limit=2,
    single_prompt=generatePrompt
)

for item in response.objects:
    print(json.dumps(item.properties, indent=1))
    print("-----vvvvvv-----")
    print(item.generated)
    print("-----^^^^^^-----")
```

### Grouped Result

Grouped Result generates a single response from all the search results. 

The below example is creating a Facebook ad from the 2 retrieved Jeoprady questions about animals. 

```python
generateTask = "Explain why these Jeopardy questions are under the Animals category."

jeopardy = client.collections.get("JeopardyQuestion")
response = jeopardy.generate.near_text(
    query="Animals",
    limit=3,
    grouped_task=generateTask
)

print(response.generated)
```