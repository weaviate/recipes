import weaviate

client = weaviate.Client(
  url="https://some-endpoint.weaviate.network",  # URL of your Weaviate instance
  auth_client_secret=weaviate.AuthApiKey(api_key="6Pk4T1WUvvUNPBbmd0Umdr38fuv6KBtwCRD3"), # (Optional) If the Weaviate instance requires authentication
  additional_headers={
        "X-Cohere-Api-Key": "YOUR-COHERE-API-KEY"  # Replace with your Cohere key
  }
)

client.schema.get()  # Get the schema to test connection