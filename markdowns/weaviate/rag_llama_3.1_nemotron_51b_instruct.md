---
layout: recipe
toc: True
title: "Generate new content with NVIDIA models and RAG"
featured: False
integration: False
agent: False
tags: ['Generative Search', 'RAG', 'NVIDIA']
---
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weaviate/recipes/blob/main/weaviate-features/model-providers/nvidia/rag_llama_3.1_nemotron_51b_instruct.ipynb)

# Generative Search with NVIDIA

In this demo, we will use an embedding and generative model on NVIDIA to generate embeddings for the blog posts and use a generative model to create new content!

## Requirements

1. Weaviate cluster
    1. You can create a 14-day free sandbox on [WCD](https://console.weaviate.cloud/)
    2. [Embedded Weaviate](https://docs.weaviate.io/deploy/installation-guides/embedded)
    3. [Local deployment](https://docs.weaviate.io/deploy/installation-guides/docker-installation)
    4. [Other options](https://docs.weaviate.io/deploy)

2. NVIDIA NIM API key. Grab one [here](https://build.nvidia.com/models).
3. Weaviate client version `4.11.0` or newer
4. Weaviate database version `1.28.5`, `1.29.0`, or newer.

```python
import weaviate
from weaviate.embedded import EmbeddedOptions
import weaviate.classes as wvc
import weaviate.classes.config as wc
import requests, json
import weaviate.classes.query as wq
from weaviate.classes.config import Property, DataType
import os
import re
from weaviate.util import get_valid_uuid
from uuid import uuid4
```

## Connect to Weaviate

Only choose one option from the below.

**Weaviate Cloud Deployment**

```python
WCD_URL = os.environ["WEAVIATE_URL"] # Replace with your Weaviate cluster URL
WCD_AUTH_KEY = os.environ["WEAVIATE_AUTH"] # Replace with your cluster auth key
NVIDIA_KEY = os.environ["NVIDIA_API_KEY"] # Replace with your NVIDIA key

# Weaviate Cloud Deployment
client = weaviate.connect_to_wcs(
    cluster_url=WCD_URL,
    auth_credentials=weaviate.auth.AuthApiKey(WCD_AUTH_KEY),
      headers={ "X-Nvidia-Api-Key": NVIDIA_KEY}
)

print(client.is_ready())
```

Python output:
```text
True
```
**Embedded Weaviate**

```python
# NVIDIA_KEY = os.environ["NVIDIA_API_KEY"] # Replace with your NVIDIA key

# client = weaviate.WeaviateClient(
#     embedded_options=EmbeddedOptions(
#         version="1.29.0",
#         additional_env_vars={
#             "ENABLE_MODULES": "text2vec-nvidia, generative-nvidia"
#         }),
#         additional_headers={
#             "X-Nvidia-Api-Key": NVIDIA_KEY
#         }
# )

# client.connect()
```

**Local Deployment**

```python
# NVIDIA_KEY = os.environ["NVIDIA_API_KEY"] # Replace with your NVIDIA key

# client = weaviate.connect_to_local(
#   headers={
#     "X-NVIDIA-Api-Key": NVIDIA_KEY
#   }
# )
# print(client.is_ready())
```

## Create a collection
Collection stores your data and vector embeddings.

Full list of [generative models](https://weaviate.io/developers/weaviate/model-providers/octoai/generative#available-models)

```python
# Note: in practice, you shouldn't rerun this cell, as it deletes your data
# in "BlogChunks", and then you need to re-import it again.

# Delete the collection if it already exists
if (client.collections.exists("BlogChunks")):
    client.collections.delete("BlogChunks")

client.collections.create(
    name="BlogChunks",

    vectorizer_config=wc.Configure.Vectorizer.text2vec_nvidia( # specify the vectorizer and model
        model="nvidia/nv-embed-v1", # optional, default is nvidia/nv-embed-v1
    ),

    generative_config=wc.Configure.Generative.nvidia( # specify the generative model
        model="nvidia/llama-3.1-nemotron-51b-instruct" # optional, default is nvidia/llama-3.1-nemotron-51b-instruct
    ),

    properties=[
            Property(name="content", data_type=DataType.TEXT) # We only have one property for our collection. It is the content within the blog posts
    ]
)

print("Successfully created collection: BlogChunks.")
```

Python output:
```text
Successfully created collection: BlogChunks.
```
## Chunk and Import Data

We need to break our blog posts into smaller chunks

```python
def chunk_list(lst, chunk_size):
    """Break a list into chunks of the specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def split_into_sentences(text):
    """Split text into sentences using regular expressions."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?&lt;=\.|\?)\s', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def read_and_chunk_index_files(main_folder_path):
    """Read index.md files from subfolders, split into sentences, and chunk every 5 sentences."""
    blog_chunks = []
    
    for file_path in os.listdir("./data"):
        index_file_path = os.path.join("./data", file_path)
        with open(index_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            sentences = split_into_sentences(content)
            sentence_chunks = chunk_list(sentences, 5)
            sentence_chunks = [' '.join(chunk) for chunk in sentence_chunks]
            blog_chunks.extend(sentence_chunks)
    return blog_chunks

# run with:
main_folder_path = './data'
blog_chunks = read_and_chunk_index_files(main_folder_path)
```

```python
# First chunk

blog_chunks[0]
```

Python output:
```text
'---\ntitle: What is Ref2Vec and why you need it for your recommendation system\nslug: ref2vec-centroid\nauthors: [connor]\ndate: 2022-11-23\ntags: [\'integrations\', \'concepts\']\nimage: ./img/hero.png\ndescription: "Weaviate introduces Ref2Vec, a new module that utilises Cross-References for Recommendation!"\n---\n![Ref2vec-centroid](https://raw.githubusercontent.com/weaviate/recipes/refs/heads/main/weaviate-features/model-providers/nvidia/img/hero.png)\n\n<!-- truncate -->\n\nWeaviate 1.16 introduced the [Ref2Vec](/developers/weaviate/modules/retriever-vectorizer-modules/ref2vec-centroid) module. In this article, we give you an overview of what Ref2Vec is and some examples in which it can add value such as recommendations or representing long objects. ## What is Ref2Vec? The name Ref2Vec is short for reference-to-vector, and it offers the ability to vectorize a data object with its cross-references to other objects. The Ref2Vec module currently holds the name ref2vec-**centroid** because it uses the average, or centroid vector, of the cross-referenced vectors to represent the **referencing** object.'
```
```python
# Insert the objects (chunks) into the Weaviate cluster

blogs = client.collections.get("BlogChunks")

for blog_chunk in blog_chunks:
    random_uuid = get_valid_uuid(uuid4())
    blogs.data.insert(
        properties={
            "content": blog_chunk
        },
        uuid=random_uuid
    )

```

## Query Time 

## Hybrid Search Query

Hybrid search combines BM25 and vector search and weighs the two algorithms depending on the `alpha` parameter. 

`alpha`= 0 --> pure BM25

`alpha`= 0.5 --> half BM25, half vector search

`alpha`= 1 --> pure vector search

```python
import json 

blogs = client.collections.get("BlogChunks")

response = blogs.query.hybrid(
    query="What is Ref2Vec",
    alpha=0.5,
    limit=3
)

for o in response.objects:
    print(json.dumps(o.properties, indent=2))
```

Python output:
```text
{
  "content": "---\ntitle: What is Ref2Vec and why you need it for your recommendation system\nslug: ref2vec-centroid\nauthors: [connor]\ndate: 2022-11-23\ntags: ['integrations', 'concepts']\nimage: ./img/hero.png\ndescription: \"Weaviate introduces Ref2Vec, a new module that utilises Cross-References for Recommendation!\"\n---\n![Ref2vec-centroid](https://raw.githubusercontent.com/weaviate/recipes/refs/heads/main/weaviate-features/model-providers/nvidia/img/hero.png)\n\n<!-- truncate -->\n\nWeaviate 1.16 introduced the [Ref2Vec](/developers/weaviate/modules/retriever-vectorizer-modules/ref2vec-centroid) module. In this article, we give you an overview of what Ref2Vec is and some examples in which it can add value such as recommendations or representing long objects. ## What is Ref2Vec? The name Ref2Vec is short for reference-to-vector, and it offers the ability to vectorize a data object with its cross-references to other objects. The Ref2Vec module currently holds the name ref2vec-**centroid** because it uses the average, or centroid vector, of the cross-referenced vectors to represent the **referencing** object."
}
{
  "content": "As you have seen above, we think Ref2Vec can add value for use cases such as recommendations, re-ranking, overcoming the cold start problem and representing long objects. We are also excited to see what you build with Ref2Vec, and excited to build on this module with its future iterations. Speaking of which, we have another blog post coming soon on the development directions of Ref2Vec for the future. We will discuss topics such as **collaborative filtering**, **multiple centroids**, **graph neural networks**, and more on **re-ranking** with Ref2Vec. Stay tuned!\n\n\nimport WhatNext from '/_includes/what-next.mdx'\n\n<WhatNext />"
}
{
  "content": "Although all the query does is provide the ID of the User object, Ref2Vec has done the hard work by inferring a centroid vector from the User's references to other vectors. And as the set of references continues to evolve, the Ref2Vec vectors will continue to evolve also, ensuring that the User vector remains up-to-date with their latest interests. Whether your goal is to construct a Home Feed interface for users or to pair with search queries, Ref2Vec provides a strong foundation to enable Recommendation with fairly low overhead. For example, it can achieve personalized re-ranking, also known as a session-based recommendation, without persisting user data over a long sequence of interactions. A new user could have personalization available after a few interactions on the app which will help them quickly settle in and feel at home, helping to overcome what is known as the cold-start problem."
}
```
### Generative Search Query

Here is what happens in the below:
1. We will retrieve 3 relevant chunks from our vector database
2. We will pass the 3 chunks to NVIDIA to generate the short paragraph about Ref2Vec

The first line in the output is the generated text, and the `content` pieces below it, are what was retrieved from Weaviate and passed to NVIDIA.

```python
blogs = client.collections.get("BlogChunks")

response = blogs.generate.near_text(
    query="What is Ref2Vec?",
    single_prompt="Write a short paragraph about ref2vec with this content: {content}",
    limit=3
)

for o in response.objects:
    print(o.generated)
    print(json.dumps(o.properties, indent=2))
```

Python output:
```text
Here is a short paragraph about Ref2Vec:

Ref2Vec, short for reference-to-vector, is a module in Weaviate 1.16 that enables the vectorization of a data object by incorporating its cross-references to other objects. The Ref2Vec module, specifically ref2vec-centroid, generates a representation of the referencing object by using the average, or centroid vector, of the cross-referenced vectors. This innovative approach can add significant value to applications such as recommendation systems and representing long objects.
{
  "content": "---\ntitle: What is Ref2Vec and why you need it for your recommendation system\nslug: ref2vec-centroid\nauthors: [connor]\ndate: 2022-11-23\ntags: ['integrations', 'concepts']\nimage: ./img/hero.png\ndescription: \"Weaviate introduces Ref2Vec, a new module that utilises Cross-References for Recommendation!\"\n---\n![Ref2vec-centroid](https://raw.githubusercontent.com/weaviate/recipes/refs/heads/main/weaviate-features/model-providers/nvidia/img/hero.png)\n\n<!-- truncate -->\n\nWeaviate 1.16 introduced the [Ref2Vec](/developers/weaviate/modules/retriever-vectorizer-modules/ref2vec-centroid) module. In this article, we give you an overview of what Ref2Vec is and some examples in which it can add value such as recommendations or representing long objects. ## What is Ref2Vec? The name Ref2Vec is short for reference-to-vector, and it offers the ability to vectorize a data object with its cross-references to other objects. The Ref2Vec module currently holds the name ref2vec-**centroid** because it uses the average, or centroid vector, of the cross-referenced vectors to represent the **referencing** object."
}
Here is a rewritten paragraph about Ref2Vec:

Weaviate's Ref2Vec technology provides a novel way to vectorize classes, like the User class, by leveraging their relationships with other classes. This enables Weaviate to rapidly generate accurate and up-to-date representations of users based on recent interactions. For instance, if a user clicks on multiple shoe images in an e-commerce store, it's likely they're interested in seeing more shoes. Ref2Vec effectively captures this intuition by generating vectors that aggregate each User's interactions with other classes. The adjacent animation illustrates this concept in action, using real-world e-commerce image interactions as an example.
{
  "content": "![Cross-reference](https://raw.githubusercontent.com/weaviate/recipes/refs/heads/main/weaviate-features/model-providers/nvidia/img/Weaviate-Ref2Vec_1.png)\n\nRef2Vec gives Weaviate another way to vectorize a class, such as the User class, based on their relationships to other classes. This allows Weaviate to quickly create up-to-date representations of users based on their relationships such as recent interactions. If a user clicks on 3 shoe images on an e-commerce store, it is a safe bet that they want to see more shoes. Ref2Vec captures this intuition by calculating vectors that aggregate each User's interaction with another class. The below animation visualizes a real example of this in e-Commerce images."
}
Here is a rewritten paragraph about Ref2Vec:

The image showcases how Ref2Vec generates a composite representation of a user by aggregating the vector embeddings of their purchased products - in this case, a pair of boots, shorts, and a Weaviate t-shirt. By leveraging these cross-references, Ref2Vec empowers Weaviate to quickly learn about user preferences and actions, enabling the creation of more accurate and up-to-date user characterizations. This enables the model to capture diverse user interests and tendencies across various dimensions, including product categories and fashion styles, resulting in highly relevant recommendations tailored to individual user preferences and style inclinations.
{
  "content": "The following image depicts how Ref2Vec aggregates the representations of 3 Product items to represent a User who has purchased a pair of boots, shorts, and Weaviate t-shirt!\n\n![Ref2Vec Image](https://raw.githubusercontent.com/weaviate/recipes/refs/heads/main/weaviate-features/model-providers/nvidia/img/ref2vec.png)\n\nSuch a representation of the User, by an aggregation of their cross-references, allows Weaviate to conveniently and immediately learn from each User's preferences and actions to provide improved and up-to-date characterizations. Ref2Vec can in other words capture each User's interests and tendencies across multiple axes, such as product categories or even fashion styles! And by doing so, the resulting recommendations can more closely match the User's product and style preferences. We envision Ref2Vec to have great potential in multiple application areas. Let's take a look at a few of them in more detail, starting with recommendation systems. ## Recommendation in Weaviate\nMany of you might primarily know Weaviate as a vector database and search engine, but Weaviate can also power high-quality, lightning-fast recommendations."
}
```