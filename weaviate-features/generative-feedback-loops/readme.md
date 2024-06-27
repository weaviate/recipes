# Generative Feedback Loops

Stay tuned for the GFL API coming to Weaviate soon!

```python
listings = weaviate_client.collections.get("WeaviateBlogChunk")

listings.data.gfl.update_property(
  view_properties=["price", "location", "pictures"],
  on_property=["description"],
  instruction="Write a compelling description of this AirBnB.",
  uuids=uuids
)
```
