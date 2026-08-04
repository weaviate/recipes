# Welcome to Weaviate Recipes 💚

![Weaviate logo](.github/Weaviate.png)

This repository covers end-to-end examples of the various features and integrations with [Weaviate](https://www.weaviate.io).

| Category | Description |
| -------------|---------|
| [Datasets](/datasets/) | Ready to use datasets to ingest data into your Weaviate cluster |
| [Integrations](/integrations)| Notebooks showing you how to use Weaviate plus another technology |
| [Weaviate Features](/weaviate-features) | Notebooks covering vector, hybrid and generative search, reranking, multi-tenancy, and more |
| [Query Agent](/query-agent/) | Notebooks showing you how to build with Weaviate's Query Agent |


## Integrations 🌐
Check out Weaviate's [Integrations Documentation](https://weaviate.io/developers/integrations)!

| Company Category | Companies |
|------------------|-----------|
| Cloud Hyperscalers | Google, AWS, NVIDIA |
| Compute Infrastructure | Modal, Replicate |
| LLM and Agent Frameworks | Agno, CrewAI, Composio, DSPy, Dynamiq, LangChain, LlamaIndex, Pydantic, Semantic Kernel, Ollama, Haystack, Modaic |
| Data Platforms| Databricks, Confluent, Box, Boomi, Spark, Unstructured, Firecrawl, Context Data, Aryn, Astronomer, Airbyte, IBM (Docling), Cardinal, Contextual AI, Chonkie, Parallel |
| Operations | AIMon, Arize, Cleanlab, Comet, DeepEval, Langtrace, LangWatch, Nomic, Patronus AI, Ragas, TruLens, Weights & Biases |


## Weaviate Features 🔧

| Feature | Description |
|---------|-------------|
| Model Providers | Use Weaviate's `nearText`, `hybrid`, and `.generate` operator with various model providers |
| Filters | Narrow down your search results by adding filters to your queries |
| Reranking | Add reranking to your pipeline to improve search results (broken out by model provider) |
| Media Search | Use Weaviate's `nearImage` and `nearVideo` operator to search using images and videos |
| Classification | Learn how to use KNN and zero-shot classification |
| Multi-Tenancy | Store tenants on separate shards for complete data isolation |
| Multi-Vector Embeddings | Use Weaviate with powerful ColBERT-style embeddings to improve search results |
| Product Quantization | Compress vector embeddings and reduce the memory footprint using Weaviate's PQ feature |
| Evaluation | Evaluate your search system |

## Query Agent 🔍
| Mode | Description |
|---------|-------------|
| Ask | Transforms your query into actionable searches or aggregations, and then provides a final answer to the question  |
| Search | Transforms your query into actionable searches and returns the matching Weaviate objects directly |
| Suggest Queries |  Suggest queries based on the data in your collection |

## Adding Recipes to Weaviate Docs

Check out [this contributor guide](./.docs/README.md) to convert recipes (Jupyter Notebooks) into docs friendly markdown.

## Feedback ❓
Please note this is an ongoing project, and updates will be made frequently. If you have a feature you would like to see, please create a GitHub issue or feel free to contribute one yourself!
