# Generative Feedback Loops

Generative Feedback Loops (GFLs) describe the co-evolution of data and AI models throughout the lifecycle of an AI application. GFLs capture the continuous exchange of feedback between data and AI models, where AI models enhance datasets by updating or creating data objects, and where data shapes AI models for specific domains or tasks. We explore the foundational relationship between data and AI models, and the dynamic nature of these interactions.

# /Update

```python
listings = weaviate_client.collections.get("WeaviateBlogChunk")

listings.data.gfl.update_property(
  view_properties=["price", "location", "pictures"],
  on_property=["description"],
  instruction="Write a compelling description of this AirBnB.",
  uuids=uuids
)
```

# /Create

```python
sneaker_designs = weaviate_client.collections.get("SneakerDesign")

sneaker_designs.gfl.create(
  view_properties=["brand_aesthetic", "season", "material_focus"],
  on_property=["Image"],
  instruction="""
  Can you generate an image of a t-shirt that would be made by {brand_name}? {brand_description}
  """,
  uuisa=uuids
)
```
