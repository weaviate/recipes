import weaviate

client = weaviate.Client(
  url="https://some-endpoint.weaviate.network",  # URL of your Weaviate instance
  auth_client_secret=weaviate.AuthApiKey(api_key="YOUR-WEAVIATE-API-KEY"), # (Optional) If the Weaviate instance requires authentication
  additional_headers={
    "X-OpenAI-Api-Key": "YOUR-OPENAI-API-KEY", # Replace with your OpenAI key
  }
)

client.schema.get()  # Get the schema to test connection