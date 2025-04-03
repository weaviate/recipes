---
layout: recipe
colab: https://colab.research.google.com/github/weaviate/recipes/blob/main//weaviate-services/agents/transformation-agent-retrieval-benchmark.ipynb
toc: True
title: "Benchmarking Arctic 2.0 vs. Arctic 1.5 with Synthetic RAG Evals"
featured: True
integration: False
agent: True
tags: ['Transformation Agent']
---
    


Traditionally, AI system evaluation has relied heavily on human-written evaluation sets. Most notably, this process demands substantial time and resource investment, preventing most developers from creating and properly evaluating their AI systems.

The Weaviate Transformation Agent offers a breakthrough in AI evaluation by enabling the rapid generation of synthetic evaluation datasets!

In this notebook, we generate **2,100 synthetic questions for the Weaviate Blogs (1 for each) in 63 seconds!**

We then use this dataset to report embedding recall (finding the source document in the haystack) between the Snowflake [Arctic 2.0](https://arxiv.org/abs/2412.04506) and [Arctic 1.5](https://arxiv.org/abs/2405.05374) embedding models. We find the following result:

| Model      | Recall@1 | Recall@5 | Recall@100 |
|------------|----------|----------|------------|
| Arctic 1.5 | 0.7995   | 0.9245   | 0.9995     |
| Arctic 2.0 | 0.8412   | 0.9546   | 0.9995     |

The Arctic 2.0 model demonstrates superior performance, with particularly notable improvements in Recall @ 1 (4.17% increase) and Recall @ 5 (3.01% increase).

## Here is an overview of what this notebook illustrates:

![Embedding Benchmark with the Transformation Agent](https://raw.githubusercontent.com/weaviate/recipes/refs/heads/main/weaviate-services/agents/images/synthetic-query-overview-new.png "Embedding Benchmark with the Transformation Agent")


```python
import weaviate
import os
from weaviate.classes.init import Auth
import weaviate.classes.config as wvcc
import re
from weaviate.util import get_valid_uuid
from uuid import uuid4
```


```python
# Connect to Weaviate Client

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY)
)

print(weaviate_client.is_ready())
```

    True


## Weaviate Named Vectors

Create 2 HSNW indexes from the *same* property in Weaviate!!

On top of that, you can set different embedding models for each!

You can learn more about Weaviate Named Vectors [here](https://weaviate.io/developers/weaviate/config-refs/schema/multi-vector) and the Weaviate Embedding Service [here](https://weaviate.io/developers/wcs/embeddings)!


```python
blogs_collection = weaviate_client.collections.create(
    name="WeaviateBlogChunks",
    vectorizer_config=[
        wvcc.Configure.NamedVectors.text2vec_weaviate(
            name="content_arctic_1_5",
            model="Snowflake/snowflake-arctic-embed-m-v1.5",
            source_properties=["content"],
        ),
        wvcc.Configure.NamedVectors.text2vec_weaviate(
            name="content_arctic_2_0",
            model="Snowflake/snowflake-arctic-embed-l-v2.0",
            source_properties=["content"],
        )
    ],
    properties=[
        wvcc.Property(name="content", data_type=wvcc.DataType.TEXT),
    ]
)
```

## Simple Directory Parser to Load Weaviate's Blogs stored in Markdown Files into Memory


```python
def chunk_list(lst, chunk_size):
    """Break a list into chunks of the specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def split_into_sentences(text):
    """Split text into sentences using regular expressions."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def read_and_chunk_index_files(main_folder_path):
    """Read index.md files from subfolders, split into sentences, and chunk every 5 sentences."""
    blog_chunks = []
    for folder_name in os.listdir(main_folder_path):
        subfolder_path = os.path.join(main_folder_path, folder_name)
        if os.path.isdir(subfolder_path):
            index_file_path = os.path.join(subfolder_path, 'index.mdx')
            if os.path.isfile(index_file_path):
                with open(index_file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    sentences = split_into_sentences(content)
                    sentence_chunks = chunk_list(sentences, 5)
                    sentence_chunks = [' '.join(chunk) for chunk in sentence_chunks]
                    blog_chunks.extend(sentence_chunks)
    return blog_chunks

# Example usage
main_folder_path = "blog"
blog_chunks = read_and_chunk_index_files(main_folder_path)
```

## We have 2,160 blog chunks from Weaviate's Blog Posts!


```python
len(blog_chunks)
```




    2160



### Here is an example of one of these blog chunks:


```python
print(blog_chunks[0])
```

    ---
    title: 'Accelerating Vector Search up to +40% with Intel’s latest Xeon CPU - Emerald Rapids'
    slug: intel
    authors: [zain, asdine, john]
    date: 2024-03-26
    image: https://raw.githubusercontent.com/weaviate/recipes/refs/heads/main/weaviate-services/agents/img/hero.png
    tags: ['engineering', 'research']
    description: 'Boosting Weaviate using SIMD-AVX512, Loop Unrolling and Compiler Optimizations'
    ---
    
    ![HERO image](https://raw.githubusercontent.com/weaviate/recipes/refs/heads/main/weaviate-services/agents/img/hero.png)
    
    **Overview of Key Sections:**
    - [**Vector Distance Calculations**](#vector-distance-calculations) Different vector distance metrics popularly used in Weaviate. - [**Implementations of Distance Calculations in Weaviate**](#vector-distance-implementations) Improvements under the hood for implementation of Dot product and L2 distance metrics. - [**Intel’s 5th Gen Intel Xeon Processor, Emerald Rapids**](#enter-intel-emerald-rapids)  More on Intel's new 5th Gen Xeon processor. - [**Benchmarking Performance**](#lets-talk-numbers) Performance numbers on microbenchmarks along with simulated real-world usage scenarios. What’s the most important calculation a vector database needs to do over and over again?


## Batch Import Blogs into Weaviate


```python
blogs_collection = weaviate_client.collections.get("WeaviateBlogChunks")

import time

start_time = time.time()
with blogs_collection.batch.dynamic() as batch:
    for i, blog_chunk in enumerate(blog_chunks):
        batch.add_object(
            properties={
                "content": blog_chunk
            },
        )
        
        if (i + 1) % 500 == 0:
            elapsed_time = time.time() - start_time
            print(f"Inserted {i + 1} blog chunks... (Time elapsed: {elapsed_time:.2f} seconds)")

failed_objects = blogs_collection.batch.failed_objects
if failed_objects:
    print(f"Number of failed imports: {len(failed_objects)}")
    print(f"First failed object: {failed_objects[0]}")
```

    Inserted 500 blog chunks... (Time elapsed: 2.26 seconds)
    Inserted 1000 blog chunks... (Time elapsed: 4.21 seconds)
    Inserted 1500 blog chunks... (Time elapsed: 4.64 seconds)
    Inserted 2000 blog chunks... (Time elapsed: 6.49 seconds)



```python
# Retry uploading failed objects with proper error handling
print("Attempting to retry failed objects...")
success_count = 0
error_count = 0

for object in failed_objects:
    try:
        # Extract the properties from the BatchObject
        properties = object.object_.properties
        
        # Insert the object with proper error handling
        blogs_collection.data.insert(
            properties={
                "content": properties["content"]  # Use "content" instead of "input_persona"
            }
        )
        success_count += 1
        
        # Add a small delay to avoid overwhelming the server
        time.sleep(0.1)
        
    except Exception as e:
        print(f"Error uploading object: {e}")
        error_count += 1
        
    # Print progress every 50 objects
    if (success_count + error_count) % 50 == 0:
        print(f"Progress: {success_count + error_count}/{len(failed_objects)} objects processed")

print(f"Retry complete. Successfully uploaded {success_count} previously failed objects. {error_count} objects still failed.")
```

    Attempting to retry failed objects...
    Retry complete. Successfully uploaded 0 previously failed objects. 0 objects still failed.


## Create Simluated User Queries with the Transformation Agent

Weaviate Users may be interested in knowing what embedding model they should use for their documents.

In order to know this, you will need some dataset of questions that you might expect your knowledge base to receive.

This can now be achieved at scale super quickly (2.1K questions in **63 seconds**) with the Transformation Agent!


```python
from weaviate.agents.transformation import TransformationAgent
from weaviate.agents.classes import Operations
from weaviate.collections.classes.config import DataType

create_questions = Operations.append_property(
    property_name="predicted_user_query",
    data_type=DataType.TEXT,
    view_properties=["content"],
    instruction="Based on the content of this blog chunk, generate a thoughtful information-seeking question that a reader might have after reading this material. The question should be specific to the technical content, concepts, or use cases mentioned in the text. Make sure the question is clear, concise, and directly related to the content provided.",
)

agent = TransformationAgent(
    client=weaviate_client,
    collection="WeaviateBlogChunks",
    operations=[create_questions],
)

response = agent.update_all()

for operation in response:  # The response is a list of TransformationResponse objects
    print(agent.get_status(workflow_id=operation.workflow_id))  # Use the workflow_id to check the status of each operation
```

    {'workflow_id': 'TransformationWorkflow-e979d5f69b91575e7289ab61d559cafa', 'status': {'batch_count': 0, 'end_time': None, 'start_time': '2025-03-12 01:21:58', 'state': 'running', 'total_duration': None, 'total_items': 0}}



```python
agent.get_status(workflow_id=response[0].workflow_id)
```




    {'workflow_id': 'TransformationWorkflow-e979d5f69b91575e7289ab61d559cafa',
     'status': {'batch_count': 9,
      'end_time': '2025-03-12 01:23:01',
      'start_time': '2025-03-12 01:21:58',
      'state': 'completed',
      'total_duration': 63.097952,
      'total_items': 2160}}




```python
response
```




    [TransformationResponse(operation_name='predicted_user_query', workflow_id='TransformationWorkflow-e979d5f69b91575e7289ab61d559cafa')]



### Inspect Synthetic Queries produced from the Blog Content


```python
blogs_collection = weaviate_client.collections.get("WeaviateBlogChunks")
```


```python
results = blogs_collection.query.hybrid(
    query="What is the Weaviate Query Agent?",
    target_vector=["content_arctic_2_0"],
    return_properties=["content", "predicted_user_query"],
    limit=5
).objects

for i, result in enumerate(results, 1):
    print(f"\033[37mExample #{i}\033[0m")
    print("\033[36mBlog Content:\033[0m\n")
    print(f"\033[36m{result.properties['content']}\033[0m")
    print("\033[32mPredicted User Query:\033[0m\n")
    print(f"\033[32m{result.properties['predicted_user_query']}\033[0m")
    print()
```

    [37mExample #1[0m
    [36mBlog Content:[0m
    
    [36m---
    title: 'Introducing the Weaviate Query Agent'
    slug: query-agent
    authors: [charles-pierse, tuana, alvin]
    date: 2025-03-05
    tags: ['concepts', 'agents', 'release']
    image: https://raw.githubusercontent.com/weaviate/recipes/refs/heads/main/weaviate-services/agents/img/hero.png
    description: "Learn about the Query Agent, our new agentic search service that redefines how you interact with Weaviate’s database!"
    ---
    ![Introducing the Weaviate Query Agent](https://raw.githubusercontent.com/weaviate/recipes/refs/heads/main/weaviate-services/agents/img/hero.png)
    
    
    We’re incredibly excited to announce that we’ve released a brand new service for our [Serverless Weaviate Cloud](https://weaviate.io/deployment/serverless) users (including free Sandbox users) to preview, currently in Alpha: the _**Weaviate Query Agent!**_ Ready to use now, this new feature provides a simple interface for users to ask complex multi-stage questions about your data in Weaviate, using powerful foundation LLMs. In this blog, learn more about what the Weaviate Query Agent is, and discover how you can build your own!
    
    Let’s get started. :::note 
    This blog comes with an accompanying [recipe](https://github.com/weaviate/recipes/tree/main/weaviate-services/agents/query-agent-get-started.ipynb) for those of you who’d like to get started. :::
    
    ## What is the Weaviate Query Agent
    
    AI Agents are semi- or fully- autonomous systems that make use of LLMs as the brain of the operation. This allows you to build applications that are able to handle complex user queries that may need to access multiple data sources. And, over the past few years we’ve started to build such applications thanks to more and more powerful LLMs capable of function calling, frameworks that simplify the development process and more.[0m
    [32mPredicted User Query:[0m
    
    [32mHow can I integrate the Weaviate Query Agent with my existing application to access multiple data sources using LLMs?[0m
    
    [37mExample #2[0m
    [36mBlog Content:[0m
    
    [36m:::note 
    To learn more about what AI Agents are, read our blog [”Agents Simplified: What we mean in the context of AI”](https://weaviate.io/blog/ai-agents). :::
    
    **With the Query Agent, we aim to provide an agent that is inherently capable of handling complex queries over multiple Weaviate collections.** The agent understands the structure of all of your collections, so knows when to run searches, aggregations or even both at the same time for you. Often, AI agents are described as LLMs that have access to various tools (adding more to its capabilities), which are also able to make a plan, and reason about the response. Our Query Agent is an AI agent that is provided access to multiple Weaviate collections within a cluster. Depending on the user’s query, it will be able to decide which collection or collections to perform searches on.[0m
    [32mPredicted User Query:[0m
    
    [32mHow does the Query Agent decide which Weaviate collections to perform searches on based on a user's query?[0m
    
    [37mExample #3[0m
    [36mBlog Content:[0m
    
    [36mSo, you can think of the Weaviate Query Agent as an AI agent that has tools in the form of Weaviate Collections. In addition to access to multiple collections, the Weaviate Query Agent also has access to two internal agentic search workflows:
    
    -   Regular [semantic search](/blog/vector-search-explained) with optional filters
    -   Aggregations
    
    In essence, we’ve released a multi-agent system that can route queries to one or the other and synthesise a final answer for the user. ![Query Agent](https://raw.githubusercontent.com/weaviate/recipes/refs/heads/main/weaviate-services/agents/img/query-agent.png)
    
    ### Routing to Search vs Aggregations
    
    Not all queries are the same. While some may require us to do semantic search using embeddings over a dataset, other queries may require us to make [aggregations](https://weaviate.io/developers/weaviate/api/graphql/aggregate) (such as counting objects, calculating the average value of a property and so on). We can demonstrate the difference with a simple example.[0m
    [32mPredicted User Query:[0m
    
    [32mHow do I decide when to use semantic search versus aggregations in the Weaviate Query Agent?[0m
    
    [37mExample #4[0m
    [36mBlog Content:[0m
    
    [36mThis allows you to provide the agent with instructions on how it should behave. For example, below we provide a system prompt which instructs the agent to always respond in the users language:
    
    ```python
    multi_lingual_agent = QueryAgent(
        client=client, collections=["Ecommerce", "Brands"],
        system_prompt="You are a helpful assistant that always generated the final response in the users language."
        " You may have to translate the user query to perform searches. But you must always respond to the user in their own language."
    )
    ```
    
    ## Summary
    
    The Weaviate Query Agent represents a significant step forward in making vector databases more accessible and powerful. By combining the capabilities of LLMs with Weaviate's own search and aggregation features, we've created a tool that can handle complex queries across multiple collections while maintaining context and supporting multiple languages. The resulting agent can be used on its own, as well as within a larger agentic or multi-agent application.[0m
    [32mPredicted User Query:[0m
    
    [32mHow do I integrate the Weaviate Query Agent with a larger agentic or multi-agent application?[0m
    
    [37mExample #5[0m
    [36mBlog Content:[0m
    
    [36mWhether you're building applications that require semantic search, complex aggregations, or both, the Query Agent simplifies the development process while providing the flexibility to customize its behavior through system prompts. As we continue to develop and enhance this feature, we look forward to seeing how our community will leverage it to build even more powerful AI-driven applications. Ready to get started? Check out our [recipe](https://colab.research.google.com/github/weaviate/recipes/blob/main/weaviate-services/agents/query-agent-get-started.ipynb), join the discussion in our forum, and start building with the Weaviate Query Agent today!
    
    import WhatsNext from '/_includes/what-next.mdx'
    
    <WhatsNext />[0m
    [32mPredicted User Query:[0m
    
    [32mHow can I customize the behavior of the Query Agent through system prompts?[0m
    


### The Generated Dataset can be found on HuggingFace [here](https://huggingface.co/datasets/weaviate/weaviate-blogs-with-synthetic-questions)!

## Benchmark Recall at 1, 5, and 100 (Arctic 1.5 vs. Arctic 2.0)


```python
recall_at_1_arctic_1_5, recall_at_1_arctic_2_0 = [], []
recall_at_5_arctic_1_5, recall_at_5_arctic_2_0 = [], []
recall_at_100_arctic_1_5, recall_at_100_arctic_2_0 = [], []

# Add counters for logging
total_items = 0
log_interval = 500

print(f"Starting evaluation...")

for item in blogs_collection.iterator():
    gold_id = item.uuid
    
    # Query with Arctic 1.5 model
    arctic_1_5_results = blogs_collection.query.hybrid(
        query=item.properties["predicted_user_query"],
        target_vector=["content_arctic_1_5"],
        limit=100
    ).objects
    
    # Query with Arctic 2.0 model
    arctic_2_0_results = blogs_collection.query.hybrid(
        query=item.properties["predicted_user_query"],
        target_vector=["content_arctic_2_0"],
        limit=100
    ).objects
    
    # Extract IDs for easier comparison
    arctic_1_5_ids = [result.uuid for result in arctic_1_5_results]
    arctic_2_0_ids = [result.uuid for result in arctic_2_0_results]
    
    # Calculate recall@1 (if the gold ID is the first result)
    recall_at_1_arctic_1_5.append(1 if arctic_1_5_ids and arctic_1_5_ids[0] == gold_id else 0)
    recall_at_1_arctic_2_0.append(1 if arctic_2_0_ids and arctic_2_0_ids[0] == gold_id else 0)
    
    # Calculate recall@5 (if the gold ID is in the first 5 results)
    recall_at_5_arctic_1_5.append(1 if gold_id in arctic_1_5_ids[:5] else 0)
    recall_at_5_arctic_2_0.append(1 if gold_id in arctic_2_0_ids[:5] else 0)
    
    # Calculate recall@100 (if the gold ID is in the results at all)
    recall_at_100_arctic_1_5.append(1 if gold_id in arctic_1_5_ids else 0)
    recall_at_100_arctic_2_0.append(1 if gold_id in arctic_2_0_ids else 0)
    
    # Increment counter
    total_items += 1
    
    # Log progress at specified intervals
    if total_items % log_interval == 0:
        # Calculate current metrics
        current_recall_at_1_arctic_1_5 = sum(recall_at_1_arctic_1_5) / len(recall_at_1_arctic_1_5)
        current_recall_at_1_arctic_2_0 = sum(recall_at_1_arctic_2_0) / len(recall_at_1_arctic_2_0)
        
        current_recall_at_5_arctic_1_5 = sum(recall_at_5_arctic_1_5) / len(recall_at_5_arctic_1_5)
        current_recall_at_5_arctic_2_0 = sum(recall_at_5_arctic_2_0) / len(recall_at_5_arctic_2_0)
        
        print(f"Processed {total_items} items...")
        print(f"  Current Arctic 1.5 - Recall@1: {current_recall_at_1_arctic_1_5:.4f}, Recall@5: {current_recall_at_5_arctic_1_5:.4f}")
        print(f"  Current Arctic 2.0 - Recall@1: {current_recall_at_1_arctic_2_0:.4f}, Recall@5: {current_recall_at_5_arctic_2_0:.4f}")

print(f"Evaluation complete! Processed {total_items} total items.")

# Calculate average recall metrics
avg_recall_at_1_arctic_1_5 = sum(recall_at_1_arctic_1_5) / len(recall_at_1_arctic_1_5) if recall_at_1_arctic_1_5 else 0
avg_recall_at_1_arctic_2_0 = sum(recall_at_1_arctic_2_0) / len(recall_at_1_arctic_2_0) if recall_at_1_arctic_2_0 else 0

avg_recall_at_5_arctic_1_5 = sum(recall_at_5_arctic_1_5) / len(recall_at_5_arctic_1_5) if recall_at_5_arctic_1_5 else 0
avg_recall_at_5_arctic_2_0 = sum(recall_at_5_arctic_2_0) / len(recall_at_5_arctic_2_0) if recall_at_5_arctic_2_0 else 0

avg_recall_at_100_arctic_1_5 = sum(recall_at_100_arctic_1_5) / len(recall_at_100_arctic_1_5) if recall_at_100_arctic_1_5 else 0
avg_recall_at_100_arctic_2_0 = sum(recall_at_100_arctic_2_0) / len(recall_at_100_arctic_2_0) if recall_at_100_arctic_2_0 else 0

# Print final results
print(f"FINAL RESULTS:")
print(f"Arctic 1.5 - Recall@1: {avg_recall_at_1_arctic_1_5:.4f}, Recall@5: {avg_recall_at_5_arctic_1_5:.4f}, Recall@100: {avg_recall_at_100_arctic_1_5:.4f}")
print(f"Arctic 2.0 - Recall@1: {avg_recall_at_1_arctic_2_0:.4f}, Recall@5: {avg_recall_at_5_arctic_2_0:.4f}, Recall@100: {avg_recall_at_100_arctic_2_0:.4f}")

```

    Starting evaluation...
    Processed 500 items...
      Current Arctic 1.5 - Recall@1: 0.8140, Recall@5: 0.9380
      Current Arctic 2.0 - Recall@1: 0.8780, Recall@5: 0.9760
    Processed 1000 items...
      Current Arctic 1.5 - Recall@1: 0.8040, Recall@5: 0.9290
      Current Arctic 2.0 - Recall@1: 0.8630, Recall@5: 0.9660
    Processed 1500 items...
      Current Arctic 1.5 - Recall@1: 0.8027, Recall@5: 0.9287
      Current Arctic 2.0 - Recall@1: 0.8493, Recall@5: 0.9567
    Processed 2000 items...
      Current Arctic 1.5 - Recall@1: 0.7980, Recall@5: 0.9235
      Current Arctic 2.0 - Recall@1: 0.8390, Recall@5: 0.9540
    Evaluation complete! Processed 2160 total items.
    FINAL RESULTS:
    Arctic 1.5 - Recall@1: 0.7995, Recall@5: 0.9245, Recall@100: 0.9995
    Arctic 2.0 - Recall@1: 0.8412, Recall@5: 0.9546, Recall@100: 0.9995


## Arctic Embed
Here are some more resources to learn about the Snowflake Arctic Embedding models!


• [Arctic Embed on GitHub](https://github.com/Snowflake-Labs/arctic-embed) <br />
• [Arctic Embed 2.0 Research Report](https://arxiv.org/abs/2412.04506) <br />
• [Arctic Embed Research Report](https://arxiv.org/abs/2405.05374) <br />
• [Weaviate Podcast #110 with Luke Merrick, Puxuan Yu, and Charles Pierse!](https://www.youtube.com/watch?v=Kjqv4uk3RCs)

## Massive thanks to Luke and Puxuan for reviewing this notebook!

![Arctic Embed on the Weaviate Podcast](https://raw.githubusercontent.com/weaviate/recipes/refs/heads/main/weaviate-services/agents/images/pod-110-thumbnail.png "Arctic Embed on the Weaviate Podcast!")
