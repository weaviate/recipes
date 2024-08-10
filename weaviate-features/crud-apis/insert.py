import weaviate
from weaviate.util import get_valid_uuid
from uuid import uuid4

weaviate_client = weaviate.connect_to_local()

code_collection = weaviate_client.collections.get("Code")

id = get_valid_uuid(uuid4())

code_collection.insert(
  properties=properties,
  uuid=id
)
