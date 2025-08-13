# Welcome to Weaviate Recipes 💚

![Weaviate logo](.github/Weaviate.png)

This repository covers end-to-end examples of the various features and integrations with [Weaviate](https://www.weaviate.io).

| Category | Description |
| -------------|---------|
| [Datasets](/datasets/) | Ready to use datasets to ingest data into your Weaviate cluster |
| [Integrations](/integrations)| Notebooks showing you how to use Weaviate plus another technology |
| [Weaviate Features](/weaviate-features) | Notebooks covering vector, hybrid and generative search, reranking, multi-tenancy, and more |
| [Weaviate Services](/weaviate-services/) | Notebooks showing you how to build with Weaviate Services |


## Integrations 🌐
Check out Weaviate's [Integrations Documentation](https://weaviate.io/developers/integrations)!

| Company Category | Companies |
|------------------|-----------|
| Cloud Hyperscalers | Google, AWS, NVIDIA |
| Compute Infrastructure | Modal, Replicate |
| LLM and Agent Frameworks | Agno, CrewAI, Composio, DSPy, Dynamiq, LangChain, LlamaIndex, Pydantic, Semantic Kernel, Ollama, Haystack |
| Data Platforms| Databricks, Confluent, Box, Boomi, Spark, Unstructured, Firecrawl, Context Data, Aryn, Astronomer, Airbyte, IBM (Docling) |
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

## Weaviate Services 🧰
| Service | Description |
|---------|-------------|
| Agents | Use Weaviate's inherent agents like the `QueryAgent` & `TransformationAgent` |
| Weaviate Embeddings | [Weaviate Embeddings](https://weaviate.io/developers/wcs/embeddings) enables you to generate embeddings directly from a [Weaviate Cloud](https://console.weaviate.cloud/) database instance. | 

## Contribution Guidelines for Adding Recipe to Weaviate Website

Requirement: For a recipe to be added to the website, it should be a Python notebook

In your PR, you should make sure you've completed the following steps:

1. Add an entry for your recipe into `index.toml`. Add any relevant and optional tags: `tags: a list of topic tags`, `agent: true or false`, `integration: true or false` etc.
2. `pip install -r requirements.txt`
3. `python scripts/generate_markdowns.py`: This will generate a markdown version of your recipe including the frontmatter needed for the website. Check that the markdown looks correct and fix any errors that running this script causes.
4. Create the PR which should incliude: your recipe, an edit to the `index.toml` as well as the genreated markdown in `markdowns/`

> 🌅 Handling images: If your recipe has an image displayed in it, we recommend embedding the image via it's github url.

> ⚠️ If you are making changed to the `recipes` repository layout, please make sure you've corrected the recipe paths in `index.toml` if they have changed location!

## Feedback ❓
Please note this is an ongoing project, and updates will be made frequently. If you have a feature you would like to see, please create a GitHub issue or feel free to contribute one yourself!
