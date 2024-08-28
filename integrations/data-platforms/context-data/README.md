
## About VectorETL
VectorETL by [Context Data](https://contextdata.ai) is a modular no-code Python framework designed to help AI and Data Engineers to:
- Quickly extract data from multiple data sources (databases, cloud storage, and local files)
- Embed using major models (including OpenAI, Cohere, and Google Gemini)
- Write to all major vector vector databases (Weaviate et al)

## How to run
1. Install VectorETL using `pip install vector-etl`
2. Configure a yaml configuration file (see example below)

```yaml
source:
  source_data_type: "Local File"
  file_path: "./customers-100.csv"
  file_type: "csv"
  chunk_size: 1000
  chunk_overlap: 0

embedding:
  embedding_model: "OpenAI"
  api_key: "your-openai-key" #replace with your OpenAI API key
  model_name: "text-embedding-ada-002"

target:
  target_database: "Weaviate"
  weaviate_url: "https://your-cluster-details.weaviate.cloud" #replace with your Weaviate cluster url
  weaviate_api_key: "your-cluster-api-key" #replace with your Weaviate API keys
  class_name: "customers"

embed_columns: []
```

3. Execute your pipeline using `vector-etl -c /path/to/config.yaml`

**That's it!**

## Additional Example Configuration Files

We have included some additional configuration examples in the [examples](examples) directory. These examples illustrate how to use VectorETL with a range of data sources and embedding models.

1. **gcs-to-weaviate-using-cohere.yaml**: Extract file(s) from Google Cloud Storage, embed using Cohere and write the resulting embeddings to Weaviate
2. **postgres-to-weaviate.yaml**: Extract data from a PostgreSQL table using a defined SQL statement, embed the results using Cohere and write the resulting embeddings to Weaviate
3. **s3-to-weaviate.yaml**: Extract file(s) from Amazon S3, embed using Cohere and write the resulting embeddings to Weaviate

Happy Vectoring!
