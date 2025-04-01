---
layout: recipe
colab: https://colab.research.google.com/github/weaviate/recipes/blob/main//weaviate-services/agents/query-agent-get-started.ipynb
toc: True
title: "Build A Weaviate Query Agent - The E-Commerce Assistant"
featured: True
integration: False
agent: True
tags: ['Query Agent']
---
    


In this recipe, we will be building a simple e-commerce assistant agent with the [Weaviate Query Agent](https://weaviate.io/developers/agents). This agent will have access to a number of Weaviate collections, and will be capable of answering complex queries about brands and clothing items, accessing information from each collection.

> 📚 You can read and learn more about this service in our ["Introducing the Weaviate Query Agent"](https://weaviate.io/blog/query-agent) blog.

To get started, we've prepared a few open datasets, available on Hugging Face. The first step will be walking through how to populate your Weaviate Cloud collections.

- [**E-commerce:**](https://huggingface.co/datasets/weaviate/agents/viewer/query-agent-ecommerce) A dataset that lists clothing items, prices, brands, reviews etc.
- [**Brands:**](https://huggingface.co/datasets/weaviate/agents/viewer/query-agent-brands) A dataset that lists clothing brands and information about them such as their parent brand, child brands, average customer rating etc.

Additionally, we also have access to some other unrelated datasets which you can use to add more capabilities and variety to other agents later on in the recipe:

- [**Financial Contracts**:](https://huggingface.co/datasets/weaviate/agents/viewer/query-agent-financial-contracts) A dataset of financial contracts between indivicuals and/or companies, as well as information on the type of contract and who has authored them.
- [**Weather**:](https://huggingface.co/datasets/weaviate/agents/viewer/query-agent-weather) Daily weather information including temperature, wind speed, percipitation, pressure etc.


>[Build A Weaviate Query Agent - The E-Commerce Assistant](#scrollTo=iGfss7TuSM_n)

>>[Setting Up Weaviate & Importing Data](#scrollTo=V7eGoTMgV597)

>>>[Prepare the Collections](#scrollTo=XqbFu9xHMzEU)

>>[Set Up the Query Agent](#scrollTo=4H7h-onziDEA)

>>[Run the Query Agent](#scrollTo=XnJiTt5LiMp-)

>>>[Ask a Question](#scrollTo=XnJiTt5LiMp-)

>>>[Ask a follow up question](#scrollTo=RqsUGStI-RC_)

>>>[Search over multiple collections](#scrollTo=sl78PHYmBe2s)

>>>[Changing the System Prompt](#scrollTo=Hm32UutOC63l)

>>>[Try More Questions](#scrollTo=70lwuxf1F38d)



## 1. Setting Up Weaviate & Importing Data

To use the Weaviate Query Agent, first, create a [Weaviate Cloud](https://weaviate.io/deployment/serverless) account👇
1. [Create Serverless Weaviate Cloud account](https://weaviate.io/deployment/serverless) and setup a free [Sandbox](https://weaviate.io/developers/wcs/manage-clusters/create#sandbox-clusters)
2. Go to 'Embedding' and enable it, by default, this will make it so that we use `Snowflake/snowflake-arctic-embed-l-v2.0` as the embedding model
3. Take note of the `WEAVIATE_URL` and `WEAVIATE_API_KEY` to connect to your cluster below

> Info: We recommend using [Weaviate Embeddings](https://weaviate.io/developers/weaviate/model-providers/weaviate) so you do not have to provide any extra keys for external embedding providers.


```python
!pip install weaviate-client[agents] datasets
```


```python
import os
from getpass import getpass

if "WEAVIATE_API_KEY" not in os.environ:
  os.environ["WEAVIATE_API_KEY"] = getpass("Weaviate API Key")
if "WEAVIATE_URL" not in os.environ:
  os.environ["WEAVIATE_URL"] = getpass("Weaviate URL")
```


```python
import weaviate
from weaviate.auth import Auth

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.environ.get("WEAVIATE_URL"),
    auth_credentials=Auth.api_key(os.environ.get("WEAVIATE_API_KEY")),
)
```

### Prepare the Collections

In the following code blocks, we are pulling our demo datasets from Hugging Face and writing them to new collections in our Weaviate Serverless cluster.

> ❗️ The `QueryAgent` uses the descriptions of collections and properties to decide which ones to use when solving queries, and to access more information about properties. You can experiment with changing these descriptions, providing more detail, and more. It's good practice to provide property descriptions too. For example, below we make sure that the `QueryAgent` knows that prices are all in USD, which is information that would otherwise be unavailable.


```python
from weaviate.classes.config import Configure, Property, DataType

# To re-run cell you may have to delete collections
# client.collections.delete("Brands")
client.collections.create(
    "Brands",
    description="A dataset that lists information about clothing brands, their parent companies, average rating and more.",
    vectorizer_config=Configure.Vectorizer.text2vec_weaviate()
)

# client.collections.delete("Ecommerce")
client.collections.create(
    "Ecommerce",
    description="A dataset that lists clothing items, their brands, prices, and more.",
    vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
    properties=[
        Property(name="collection", data_type=DataType.TEXT),
        Property(name="category", data_type=DataType.TEXT),
        Property(name="tags", data_type=DataType.TEXT_ARRAY),
        Property(name="subcategory", data_type=DataType.TEXT),
        Property(name="name", data_type=DataType.TEXT),
        Property(name="description", data_type=DataType.TEXT),
        Property(name="brand", data_type=DataType.TEXT),
        Property(name="product_id", data_type=DataType.UUID),
        Property(name="colors", data_type=DataType.TEXT_ARRAY),
        Property(name="reviews", data_type=DataType.TEXT_ARRAY),
        Property(name="image_url", data_type=DataType.TEXT),
        Property(name="price", data_type=DataType.NUMBER, description="price of item in USD"),
    ]
)

# client.collections.delete("Weather")
client.collections.create(
    "Weather",
    description="Daily weather information including temperature, wind speed, percipitation, pressure etc.",
    vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
    properties=[
        Property(name="date", data_type=DataType.DATE),
        Property(name="humidity", data_type=DataType.NUMBER),
        Property(name="precipitation", data_type=DataType.NUMBER),
        Property(name="wind_speed", data_type=DataType.NUMBER),
        Property(name="visibility", data_type=DataType.NUMBER),
        Property(name="pressure", data_type=DataType.NUMBER),
        Property(name="temperature", data_type=DataType.NUMBER, description="temperature value in Celsius")
    ]
)

# client.collections.delete("Financial_contracts")
client.collections.create(
    "Financial_contracts",
    description="A dataset of financial contracts between indivicuals and/or companies, as well as information on the type of contract and who has authored them.",
    vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
)
```


```python
from datasets import load_dataset

brands_dataset = load_dataset("weaviate/agents", "query-agent-brands", split="train", streaming=True)
ecommerce_dataset = load_dataset("weaviate/agents", "query-agent-ecommerce", split="train", streaming=True)
weather_dataset = load_dataset("weaviate/agents", "query-agent-weather", split="train", streaming=True)
financial_dataset = load_dataset("weaviate/agents", "query-agent-financial-contracts", split="train", streaming=True)

brands_collection = client.collections.get("Brands")
ecommerce_collection = client.collections.get("Ecommerce")
weather_collection = client.collections.get("Weather")
financial_collection = client.collections.get("Financial_contracts")

with brands_collection.batch.dynamic() as batch:
    for item in brands_dataset:
        batch.add_object(properties=item["properties"])

with ecommerce_collection.batch.dynamic() as batch:
    for item in ecommerce_dataset:
        batch.add_object(properties=item["properties"])

with weather_collection.batch.dynamic() as batch:
    for item in weather_dataset:
        batch.add_object(properties=item["properties"])

with financial_collection.batch.dynamic() as batch:
    for item in financial_dataset:
        batch.add_object(properties=item["properties"])
```

## 2. Set Up the Query Agent

When setting up the query agent, we have to provide it a few things:
- The `client`
- The `collection` which we want the agent to have access to.
- (Optionally) A `system_prompt` that describes how our agent should behave
- (Optionally) Timeout - which for now defaults to 60s.

Let's start with a simple agent. Here, we're creating an `agent` that has access to our `Brands` & `Ecommerce` datasets.


```python
from weaviate.agents.query import QueryAgent

agent = QueryAgent(
    client=client, collections=["Ecommerce", "Brands"],
)
```

## 3. Run the Query Agent

When we run the agent, it will first make a few decisions, depending on the query:

1. The agent will decide which collection or collections to look up an answer in.
2. The agent will also decide whether to perform a regular ***search query***, what ***filters*** to use, whether to do an ***aggregation query***, or all of them together!
3. It will then provide a reponse within **`QueryAgentResponse`**. We will use the `print_query_agent_response` function for a nice display of various information provided in the response object.

### Ask a Question
**Let's start with a simple question: "I like the vintage clothes, can you list me some options that are less than $200?"**

We can then also inspect how the agent responded, what kind of searches it performed on which collections, whether it has identified if the final answer is missing information or not, as well as the final answer 👇


```python
from weaviate.agents.utils import print_query_agent_response

response = agent.run("I like the vintage clothes, can you list me some options that are less than $200?")
print_query_agent_response(response)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">╭─────────────────────────────────────────────── 🔍 Original Query ───────────────────────────────────────────────╮
│                                                                                                                 │
│ I like the vintage clothes, can you list me some options that are less than $200?                               │
│                                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">╭────────────────────────────────────────────────</span> 📝 Final Answer <span style="color: #008080; text-decoration-color: #008080">────────────────────────────────────────────────╮</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ If you are looking for vintage clothing options under $200, here are some great choices:                        │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ 1. **Vintage Philosopher Midi Dress** - Priced at $125, this dress from Echo &amp; Stitch embraces a classic        │</span>
<span style="color: #008080; text-decoration-color: #008080">│ scholarly look with its deep green velvet fabric and antique gold detailing. It's tailored for elegance and is  │</span>
<span style="color: #008080; text-decoration-color: #008080">│ ideal for sophisticated occasions.                                                                              │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ 2. **Vintage Gale Pleated Dress** - This $120 dress from Solemn Chic features deep burgundy pleats and          │</span>
<span style="color: #008080; text-decoration-color: #008080">│ vintage-inspired sleeve details, perfect for a timeless scholarly appearance.                                   │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ 3. **Retro Groove Flared Pants** - For $59, these electric blue flared pants from Vivid Verse bring back the    │</span>
<span style="color: #008080; text-decoration-color: #008080">│ playful spirit of the early 2000s with a modern touch.                                                          │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ 4. **Vintage Scholar Tote** - At $90, this tote from Echo &amp; Stitch combines functionality and elegance, ideal   │</span>
<span style="color: #008080; text-decoration-color: #008080">│ for everyday use, especially if you enjoy a scholarly aesthetic.                                                │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ 5. **Electric Velvet Trousers** - Priced at $60, these neon green velvet trousers from Vivid Verse offer a fun, │</span>
<span style="color: #008080; text-decoration-color: #008080">│ throwback vibe to early Y2K fashion.                                                                            │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ 6. **Victorian Velvet Jumpsuit** - For $120, this jumpsuit from Solemn Chic offers an elegant blend of romance  │</span>
<span style="color: #008080; text-decoration-color: #008080">│ and scholarly charm, suited for library visits or cultured gatherings.                                          │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ 7. **Vintage Scholar Turtleneck** - This $55 turtleneck from Echo &amp; Stitch suits the Dark Academia vibe,        │</span>
<span style="color: #008080; text-decoration-color: #008080">│ perfect for layering or wearing alone.                                                                          │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ 8. **Vintage Ivy Loafers** - These $120 loafers from Solemn Chic offer timeless sophistication, with a deep     │</span>
<span style="color: #008080; text-decoration-color: #008080">│ burgundy finish that complements any vintage wardrobe.                                                          │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ These options cater to various preferences, from dresses and jumpsuits to pants and accessories, all capturing  │</span>
<span style="color: #008080; text-decoration-color: #008080">│ the vintage essence at an affordable price.                                                                     │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">╭───────────────────────────────────────────</span> 🔭 Searches Executed 1/1 <span style="color: #c0c0c0; text-decoration-color: #c0c0c0">────────────────────────────────────────────╮</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│ </span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">QueryResultWithCollection</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">(</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">queries</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'vintage clothes'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">]</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                                │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">filters</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">[</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                   │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│         </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">[</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                       │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│             </span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">IntegerPropertyFilter</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">(</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                              │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                 </span><span style="color: #808000; text-decoration-color: #808000">property_name</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008000; text-decoration-color: #008000">'price'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                          │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                 </span><span style="color: #808000; text-decoration-color: #808000">operator</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">ComparisonOperator.LESS_THAN:</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #008000; text-decoration-color: #008000">'&lt;'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">&gt;</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                   │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                 </span><span style="color: #808000; text-decoration-color: #808000">value</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200.0</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                     │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│             </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">)</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                   │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│         </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">]</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                       │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">]</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                                                          │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">filter_operators</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008000; text-decoration-color: #008000">'AND'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                                     │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">collection</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008000; text-decoration-color: #008000">'Ecommerce'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│ </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">)</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                               │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #d75f5f; text-decoration-color: #d75f5f">╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #d75f5f; text-decoration-color: #d75f5f">│                                                                                                                 │</span>
<span style="color: #d75f5f; text-decoration-color: #d75f5f">│ 📊 No Aggregations Run                                                                                          │</span>
<span style="color: #d75f5f; text-decoration-color: #d75f5f">│                                                                                                                 │</span>
<span style="color: #d75f5f; text-decoration-color: #d75f5f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">╭──────────────────────────────────────────────────</span> 📚 Sources <span style="color: #c0c0c0; text-decoration-color: #c0c0c0">───────────────────────────────────────────────────╮</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='5e9c5298-5b3a-4d80-b226-64b2ff6689b7' collection='Ecommerce'                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='48896222-d098-42e6-80df-ad4b03723c19' collection='Ecommerce'                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='00b383ca-262f-4251-b513-dafd4862c021' collection='Ecommerce'                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='cbe8f8be-304b-409d-a2a1-bafa0bbf249c' collection='Ecommerce'                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='c18d3c5b-8fbe-4816-bc60-174f336a982f' collection='Ecommerce'                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='1811da1b-6930-4bd1-832e-f8fa2119d4df' collection='Ecommerce'                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='2edd1bc5-777e-4376-95cd-42a141ffb71e' collection='Ecommerce'                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='9819907c-1015-4b4c-ac75-3b3848e7c247' collection='Ecommerce'                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-style: italic">   📊 Usage Statistics   </span>
┌────────────────┬──────┐
│ LLM Requests:  │ 3    │
│ Input Tokens:  │ 7774 │
│ Output Tokens: │ 512  │
│ Total Tokens:  │ 8286 │
└────────────────┴──────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">Total Time Taken:</span><span style="color: #008080; text-decoration-color: #008080"> </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16.</span><span style="color: #008080; text-decoration-color: #008080">93s</span>
</pre>



### Ask a follow up question

The agent can also be provided with additional context. For example, we can provide the previous response as context and get a `new_response`


```python
new_response = agent.run("What about some nice shoes, same budget as before?", context=response)
print_query_agent_response(new_response)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">╭─────────────────────────────────────────────── 🔍 Original Query ───────────────────────────────────────────────╮
│                                                                                                                 │
│ What about some nice shoes, same budget as before?                                                              │
│                                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">╭────────────────────────────────────────────────</span> 📝 Final Answer <span style="color: #008080; text-decoration-color: #008080">────────────────────────────────────────────────╮</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ Here are some great shoe options under $200 that you might like:                                                │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ 1. **Vintage Noir Loafers** - Priced at $125, these loafers are part of the Dark Academia collection by Solemn  │</span>
<span style="color: #008080; text-decoration-color: #008080">│ Chic. They come in black and grey, featuring a classic design with a modern twist. Reviews highlight their      │</span>
<span style="color: #008080; text-decoration-color: #008080">│ comfort and stylish appearance, making them suitable for both casual and formal settings.                       │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ 2. **Parchment Boots** - At $145, these boots from Nova Nest's Light Academia collection are noted for their    │</span>
<span style="color: #008080; text-decoration-color: #008080">│ elegant ivory leather and classical detail stitching. They are praised for their comfort and versatile style.   │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ 3. **Bramble Berry Loafers** - These loafers, priced at $75, come in pink and green and are marked by their     │</span>
<span style="color: #008080; text-decoration-color: #008080">│ eco-friendly material and countryside aesthetic. Produced by Eko &amp; Stitch, they are loved for their comfort and │</span>
<span style="color: #008080; text-decoration-color: #008080">│ sustainability.                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ 4. **Glide Platforms** - Available for $90 from the Y2K collection by Vivid Verse, these platform sneakers are  │</span>
<span style="color: #008080; text-decoration-color: #008080">│ both comfortable and stylish with a high-shine pink finish.                                                     │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ 5. **Sky Shimmer Sneaks** - Costing $69, these sneakers are from the Y2K collection by Nova Nest and offer a    │</span>
<span style="color: #008080; text-decoration-color: #008080">│ comfortable fit with a touch of sparkle for style.                                                              │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ These selections offer a mix of formal and casual styles, ensuring you can find a perfect pair under your       │</span>
<span style="color: #008080; text-decoration-color: #008080">│ budget of $200.                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">╭───────────────────────────────────────────</span> 🔭 Searches Executed 1/1 <span style="color: #c0c0c0; text-decoration-color: #c0c0c0">────────────────────────────────────────────╮</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│ </span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">QueryResultWithCollection</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">(</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">queries</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'nice shoes'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">]</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                                     │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">filters</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">[</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                   │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│         </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">[</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                       │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│             </span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">IntegerPropertyFilter</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">(</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                              │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                 </span><span style="color: #808000; text-decoration-color: #808000">property_name</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008000; text-decoration-color: #008000">'price'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                          │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                 </span><span style="color: #808000; text-decoration-color: #808000">operator</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">ComparisonOperator.LESS_THAN:</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #008000; text-decoration-color: #008000">'&lt;'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">&gt;</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                   │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                 </span><span style="color: #808000; text-decoration-color: #808000">value</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200.0</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                     │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│             </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">)</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                   │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│         </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">]</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                       │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">]</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                                                          │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">filter_operators</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008000; text-decoration-color: #008000">'AND'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                                     │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">collection</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008000; text-decoration-color: #008000">'Ecommerce'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│ </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">)</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                               │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #d75f5f; text-decoration-color: #d75f5f">╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #d75f5f; text-decoration-color: #d75f5f">│                                                                                                                 │</span>
<span style="color: #d75f5f; text-decoration-color: #d75f5f">│ 📊 No Aggregations Run                                                                                          │</span>
<span style="color: #d75f5f; text-decoration-color: #d75f5f">│                                                                                                                 │</span>
<span style="color: #d75f5f; text-decoration-color: #d75f5f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">╭──────────────────────────────────────────────────</span> 📚 Sources <span style="color: #c0c0c0; text-decoration-color: #c0c0c0">───────────────────────────────────────────────────╮</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='96b30047-8ce1-4096-9bcf-009733cf8613' collection='Ecommerce'                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='61e4fcd7-d2bc-4861-beb6-4c16948d9921' collection='Ecommerce'                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='6e533f7d-eba1-4e74-953c-9d43008278e7' collection='Ecommerce'                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='f873ac48-1311-462a-86b2-a28b15fdda7a' collection='Ecommerce'                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='93b8b13e-a417-4be2-9cce-fda8c767f35e' collection='Ecommerce'                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-style: italic">   📊 Usage Statistics    </span>
┌────────────────┬───────┐
│ LLM Requests:  │ 4     │
│ Input Tokens:  │ 9783  │
│ Output Tokens: │ 574   │
│ Total Tokens:  │ 10357 │
└────────────────┴───────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">Total Time Taken:</span><span style="color: #008080; text-decoration-color: #008080"> </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">18.</span><span style="color: #008080; text-decoration-color: #008080">02s</span>
</pre>



Now let's try a question that sholud require an aggregation. Let's see which brand lists the most shoes.


```python
response = agent.run("What is the the name of the brand that lists the most shoes?")
print_query_agent_response(response)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">╭─────────────────────────────────────────────── 🔍 Original Query ───────────────────────────────────────────────╮
│                                                                                                                 │
│ What is the the name of the brand that lists the most shoes?                                                    │
│                                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">╭────────────────────────────────────────────────</span> 📝 Final Answer <span style="color: #008080; text-decoration-color: #008080">────────────────────────────────────────────────╮</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ The brand that lists the most shoes is Loom &amp; Aura with a total of 118 shoe listings.                           │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #d75f5f; text-decoration-color: #d75f5f">╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮</span>
<span style="color: #d75f5f; text-decoration-color: #d75f5f">│                                                                                                                 │</span>
<span style="color: #d75f5f; text-decoration-color: #d75f5f">│ 🔭 No Searches Run                                                                                              │</span>
<span style="color: #d75f5f; text-decoration-color: #d75f5f">│                                                                                                                 │</span>
<span style="color: #d75f5f; text-decoration-color: #d75f5f">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">╭────────────────────────────────────────────</span> 📊 Aggregations Run 1/1 <span style="color: #c0c0c0; text-decoration-color: #c0c0c0">────────────────────────────────────────────╮</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│ </span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">AggregationResultWithCollection</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">(</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">search_query</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                                          │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">groupby_property</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008000; text-decoration-color: #008000">'brand'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                                   │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">aggregations</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">[</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                              │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│         </span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">IntegerPropertyAggregation</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">property_name</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008000; text-decoration-color: #008000">'collection'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">, </span><span style="color: #808000; text-decoration-color: #808000">metrics</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">NumericMetrics.COUNT:</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #008000; text-decoration-color: #008000">'COUNT'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">&gt;)</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">         │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">]</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                                                          │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">filters</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">[]</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">collection</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008000; text-decoration-color: #008000">'Ecommerce'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│ </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">)</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                               │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-style: italic">   📊 Usage Statistics   </span>
┌────────────────┬──────┐
│ LLM Requests:  │ 3    │
│ Input Tokens:  │ 3976 │
│ Output Tokens: │ 159  │
│ Total Tokens:  │ 4135 │
└────────────────┴──────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">Total Time Taken:</span><span style="color: #008080; text-decoration-color: #008080"> </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5.</span><span style="color: #008080; text-decoration-color: #008080">33s</span>
</pre>



### Search over multiple collections

In some cases, we need to combine the results of searches across multiple collections. From the result above, we can see that "Loom & Aura" lists the most shoes.

Let's imagine a scenario where the user would now want to find out more about this company, _as well_ as the items that they sell.


```python
response = agent.run("Does the brand 'Loom & Aura' have a parent brand or child brands and what countries do they operate from? "
                     "Also, what's the average price of a item from 'Loom & Aura'?")

print_query_agent_response(response)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">╭─────────────────────────────────────────────── 🔍 Original Query ───────────────────────────────────────────────╮
│                                                                                                                 │
│ Does the brand 'Loom &amp; Aura' have a parent brand or child brands and what countries do they operate from? Also, │
│ what's the average price of a item from 'Loom &amp; Aura'?                                                          │
│                                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">╭────────────────────────────────────────────────</span> 📝 Final Answer <span style="color: #008080; text-decoration-color: #008080">────────────────────────────────────────────────╮</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ Loom &amp; Aura is itself a well-established brand based in Italy and operates as the parent brand to several child │</span>
<span style="color: #008080; text-decoration-color: #008080">│ brands. These child brands include 'Loom &amp; Aura Active', 'Loom &amp; Aura Kids', 'Nova Nest', 'Vivid Verse', 'Loom  │</span>
<span style="color: #008080; text-decoration-color: #008080">│ Luxe', 'Saffron Sage', 'Stellar Stitch', 'Nova Nectar', 'Canvas Core', and 'Loom Lure'. The countries           │</span>
<span style="color: #008080; text-decoration-color: #008080">│ associated with the operations or origins of these child brands include Italy, USA, UK, Spain, South Korea,     │</span>
<span style="color: #008080; text-decoration-color: #008080">│ Japan, and some extend beyond Italy as suggested by the presence of these brands in different countries.        │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">│ The average price of an item from Loom &amp; Aura is approximately $87.11. This reflects the brand's positioning as │</span>
<span style="color: #008080; text-decoration-color: #008080">│ offering items of timeless elegance and quality craftsmanship.                                                  │</span>
<span style="color: #008080; text-decoration-color: #008080">│                                                                                                                 │</span>
<span style="color: #008080; text-decoration-color: #008080">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">╭───────────────────────────────────────────</span> 🔭 Searches Executed 1/2 <span style="color: #c0c0c0; text-decoration-color: #c0c0c0">────────────────────────────────────────────╮</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│ </span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">QueryResultWithCollection</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">(</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">queries</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'parent brand of Loom &amp; Aura'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">, </span><span style="color: #008000; text-decoration-color: #008000">'child brands of Loom &amp; Aura'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">]</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                     │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">filters</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">[[]</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">, </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">[]]</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                                           │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">filter_operators</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008000; text-decoration-color: #008000">'AND'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                                     │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">collection</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008000; text-decoration-color: #008000">'Brands'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                         │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│ </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">)</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                               │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">╭───────────────────────────────────────────</span> 🔭 Searches Executed 2/2 <span style="color: #c0c0c0; text-decoration-color: #c0c0c0">────────────────────────────────────────────╮</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│ </span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">QueryResultWithCollection</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">(</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">queries</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'Loom &amp; Aura'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">]</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                                    │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">filters</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">[</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                   │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│         </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">[</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                       │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│             </span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">TextPropertyFilter</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">(</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                 </span><span style="color: #808000; text-decoration-color: #808000">property_name</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008000; text-decoration-color: #008000">'name'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                           │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                 </span><span style="color: #808000; text-decoration-color: #808000">operator</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">ComparisonOperator.LIKE:</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #008000; text-decoration-color: #008000">'LIKE'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">&gt;</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                     │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                 </span><span style="color: #808000; text-decoration-color: #808000">value</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008000; text-decoration-color: #008000">'Loom &amp; Aura'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                             │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│             </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">)</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                   │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│         </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">]</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                       │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">]</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                                                          │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">filter_operators</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008000; text-decoration-color: #008000">'AND'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                                     │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">collection</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008000; text-decoration-color: #008000">'Brands'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                         │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│ </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">)</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                               │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">╭────────────────────────────────────────────</span> 📊 Aggregations Run 1/1 <span style="color: #c0c0c0; text-decoration-color: #c0c0c0">────────────────────────────────────────────╮</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│ </span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">AggregationResultWithCollection</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">(</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">search_query</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                                          │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">groupby_property</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">aggregations</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">[</span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">IntegerPropertyAggregation</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">property_name</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008000; text-decoration-color: #008000">'price'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">, </span><span style="color: #808000; text-decoration-color: #808000">metrics</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">NumericMetrics.MEAN:</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #008000; text-decoration-color: #008000">'MEAN'</span><span style="color: #000000; text-decoration-color: #000000">&gt;</span><span style="color: #000000; text-decoration-color: #000000; font-weight: bold">)]</span><span style="color: #000000; text-decoration-color: #000000">,</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">    │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│ </span><span style="color: #000000; text-decoration-color: #000000">    </span><span style="color: #808000; text-decoration-color: #808000">filters</span><span style="color: #000000; text-decoration-color: #000000">=</span><span style="color: #000000; text-decoration-color: #000000; font-weight: bold">[</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                   │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│ </span><span style="color: #000000; text-decoration-color: #000000">        </span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">TextPropertyFilter</span><span style="color: #000000; text-decoration-color: #000000; font-weight: bold">(</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                     │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│ </span><span style="color: #000000; text-decoration-color: #000000">            </span><span style="color: #808000; text-decoration-color: #808000">property_name</span><span style="color: #000000; text-decoration-color: #000000">=</span><span style="color: #008000; text-decoration-color: #008000">'brand'</span><span style="color: #000000; text-decoration-color: #000000">,</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                              │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│ </span><span style="color: #000000; text-decoration-color: #000000">            </span><span style="color: #808000; text-decoration-color: #808000">operator</span><span style="color: #000000; text-decoration-color: #000000">=&lt;ComparisonOperator.EQUALS: </span><span style="color: #008000; text-decoration-color: #008000">'='</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">&gt;</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                          │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│             </span><span style="color: #808000; text-decoration-color: #808000">value</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008000; text-decoration-color: #008000">'Loom &amp; Aura'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│         </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">)</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                       │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">]</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">,                                                                                                          │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│     </span><span style="color: #808000; text-decoration-color: #808000">collection</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">=</span><span style="color: #008000; text-decoration-color: #008000">'Ecommerce'</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                      │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│ </span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0; font-weight: bold">)</span><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">                                                                                                               │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #c0c0c0; text-decoration-color: #c0c0c0">╭──────────────────────────────────────────────────</span> 📚 Sources <span style="color: #c0c0c0; text-decoration-color: #c0c0c0">───────────────────────────────────────────────────╮</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='88433e18-216d-489a-8719-81a29b0ae915' collection='Brands'                                         │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='99f42d07-51e9-4388-9c4b-63eb8f79f5fd' collection='Brands'                                         │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='0852c2a4-0c5a-4c69-9762-1be10bc44f2b' collection='Brands'                                         │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='d172a342-da41-45c3-876e-d08db843b8b3' collection='Brands'                                         │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='a7ad0ed7-812e-4106-a29f-40442c3a106e' collection='Brands'                                         │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│  - object_id='b6abfa02-18e5-44cf-a002-ba140e3623ad' collection='Brands'                                         │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">│                                                                                                                 │</span>
<span style="color: #c0c0c0; text-decoration-color: #c0c0c0">╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
</pre>



    
    



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-style: italic">   📊 Usage Statistics    </span>
┌────────────────┬───────┐
│ LLM Requests:  │ 5     │
│ Input Tokens:  │ 9728  │
│ Output Tokens: │ 479   │
│ Total Tokens:  │ 10207 │
└────────────────┴───────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">Total Time Taken:</span><span style="color: #008080; text-decoration-color: #008080"> </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11.</span><span style="color: #008080; text-decoration-color: #008080">38s</span>
</pre>



### Changing the System Prompt

In some cases, you may want to define a custom `system_prompt` for your agent. This can help you provide the agent with some default instructions as to how to behave. For example, let's create an agent that will always answer the query in the users language.

Let's also create a `QueryAgent` that has access to two more collections, `Financial_contracts` and `Weather`. Next, you can try out more queries yourself!


```python
multi_lingual_agent = QueryAgent(
    client=client, collections=["Ecommerce", "Brands", "Financial_contracts", "Weather"],
    system_prompt="You are a helpful assistant that always generated the final response in the users language."
    " You may have to translate the user query to perform searches. But you must always respond to the user in their own language."
)
```

For example, this time lets ask something that is about weather!


```python
response = multi_lingual_agent.run("Quelles sont les vitesses minimales, maximales et moyennes du vent?")
print(response.final_answer)
```

    Les vitesses de vent minimales, maximales et moyennes sont respectivement de 8,40 km/h, 94,88 km/h et 49,37 km/h. Ces données offrent une vue d'ensemble des conditions de vent typiques mesurées dans une période ou un lieu donné.


### Try More Questions

- For example Let's try to find out more about the brans "Eko & Stitch"


```python
response = multi_lingual_agent.run("Does Eko & Stitch have a branch in the UK? Or if not, does it have parent or child company in the UK?")

print(response.final_answer)
```

    Yes, Eko & Stitch has a branch in the UK. The brand is part of the broader company Nova Nest, which serves as Eko & Stitch's parent brand. Eko & Stitch itself operates in the UK and has its child brands, Eko & Stitch Active and Eko & Stitch Kids, also within the UK.


- Our `multi_lingual_agent` also has access to a collection called "Financial_contracts". Let's try to find out some more information about this dataser.


```python
response = multi_lingual_agent.run("What kinds of contracts are listed? What's the most common type of contract?")

print(response.final_answer)
```

    The query seeks to identify the types of contracts listed and determine the most common type. Among the types of contracts provided in the results, the following were identified: employment contracts, sales agreements, invoice contracts, service agreements, and lease agreements. The most common type of contract found in the search results is the employment contract. However, when considering data from both search and aggregation results, the aggregation reveals that the invoice contract is the most common, followed by service agreements and lease agreements. While employment contracts appear frequently in the search results, they rank fourth in the aggregation data in terms of overall occurrences.

