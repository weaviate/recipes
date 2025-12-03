import os
import weaviate
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class WeaviateClient:
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        self.url = url or os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self.api_key = api_key or os.getenv("WEAVIATE_API_KEY")
        self.client = None
        self._connect()

    def _connect(self):
        try:
            if "localhost" in self.url or self.url == "http://localhost:8080":
                if self.api_key:
                    self.client = weaviate.connect_to_local(
                        auth_credentials=weaviate.auth.AuthApiKey(api_key=self.api_key),
                        skip_init_checks=True,
                    )
                else:
                    self.client = weaviate.connect_to_local(skip_init_checks=True)
            elif "weaviate.io" in self.url or "weaviate.cloud" in self.url:
                if not self.api_key:
                    raise ConnectionError("API key is required for Weaviate Cloud")
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.url,
                    auth_credentials=weaviate.auth.AuthApiKey(api_key=self.api_key),
                    skip_init_checks=True,
                )
            else:
                url_parts = (
                    self.url.replace("http://", "").replace("https://", "").split(":")
                )
                host = url_parts[0]
                port = (
                    int(url_parts[1])
                    if len(url_parts) > 1
                    else (443 if self.url.startswith("https://") else 80)
                )
                http_secure = self.url.startswith("https://")

                if self.api_key:
                    self.client = weaviate.connect_to_custom(
                        http_host=host,
                        http_port=port,
                        http_secure=http_secure,
                        auth_credentials=weaviate.auth.AuthApiKey(api_key=self.api_key),
                        skip_init_checks=True,
                    )
                else:
                    self.client = weaviate.connect_to_custom(
                        http_host=host,
                        http_port=port,
                        http_secure=http_secure,
                        skip_init_checks=True,
                    )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Weaviate: {str(e)}")

    def reconnect(self, url: Optional[str] = None, api_key: Optional[str] = None):
        if url:
            self.url = url
        if api_key:
            self.api_key = api_key
        if self.client:
            self.client.close()
        self._connect()
        return {"status": "success", "message": f"Connected to Weaviate at {self.url}"}


weaviate_client = WeaviateClient()


def create_collection(
    class_name: str,
    description: Optional[str] = None,
    properties: Optional[List[Dict[str, Any]]] = None,
    vectorizer: Optional[str] = None,
) -> dict:
    """Create a new collection (class) in Weaviate."""
    try:
        if properties is None:
            properties = [
                weaviate.classes.Property(
                    name="text",
                    data_type=weaviate.classes.DataType.TEXT,
                    description="Text content",
                )
            ]
        else:
            props = []
            for prop in properties:
                if isinstance(prop, dict):
                    data_type_str = prop.get("dataType", ["text"])
                    if isinstance(data_type_str, list):
                        data_type_str = data_type_str[0]
                    data_type_map = {
                        "text": weaviate.classes.DataType.TEXT,
                        "string": weaviate.classes.DataType.TEXT,
                        "int": weaviate.classes.DataType.INT,
                        "number": weaviate.classes.DataType.NUMBER,
                        "boolean": weaviate.classes.DataType.BOOL,
                        "date": weaviate.classes.DataType.DATE,
                    }
                    data_type = data_type_map.get(
                        data_type_str.lower(), weaviate.classes.DataType.TEXT
                    )
                    props.append(
                        weaviate.classes.Property(
                            name=prop["name"],
                            data_type=data_type,
                            description=prop.get("description"),
                        )
                    )
                else:
                    props.append(prop)
            properties = props

        vectorizer_config = None
        if vectorizer:
            if "text2vec-openai" in vectorizer.lower():
                vectorizer_config = (
                    weaviate.classes.Configure.Vectorizer.text2vec_openai()
                )
            elif "text2vec-cohere" in vectorizer.lower():
                vectorizer_config = (
                    weaviate.classes.Configure.Vectorizer.text2vec_cohere()
                )
            elif "text2vec-huggingface" in vectorizer.lower():
                vectorizer_config = (
                    weaviate.classes.Configure.Vectorizer.text2vec_huggingface()
                )
            else:
                vectorizer_config = weaviate.classes.Configure.Vectorizer.none()

        weaviate_client.client.collections.create(
            name=class_name,
            description=description,
            properties=properties,
            vectorizer_config=vectorizer_config,
        )
        return {
            "status": "success",
            "message": f"Collection '{class_name}' created successfully",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def delete_collection(class_name: str) -> dict:
    """Delete a collection (class) from Weaviate."""
    try:
        weaviate_client.client.collections.delete(class_name)
        return {
            "status": "success",
            "message": f"Collection '{class_name}' deleted successfully",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def list_collections() -> dict:
    """List all collections (classes) in Weaviate."""
    try:
        collections = weaviate_client.client.collections.list_all()
        return {"status": "success", "collections": list(collections.keys())}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_collection_schema(class_name: str) -> dict:
    """Get the schema of a specific collection."""
    try:
        collection = weaviate_client.client.collections.get(class_name)
        config = collection.config.get()
        schema = {
            "name": config.name,
            "description": config.description,
            "properties": [
                {"name": p.name, "data_type": str(p.data_type)}
                for p in config.properties
            ],
        }
        return {"status": "success", "schema": schema}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def insert_object(
    class_name: str,
    properties: Dict[str, Any],
    uuid: Optional[str] = None,
    vector: Optional[List[float]] = None,
) -> dict:
    """Insert an object into a collection."""
    try:
        collection = weaviate_client.client.collections.get(class_name)
        result = collection.data.insert(properties=properties, uuid=uuid, vector=vector)
        return {
            "status": "success",
            "uuid": str(result),
            "message": "Object inserted successfully",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_object(class_name: str, uuid: str, include_vector: bool = False) -> dict:
    """Get an object by UUID from a collection."""
    try:
        collection = weaviate_client.client.collections.get(class_name)
        result = collection.query.fetch_object_by_id(
            uuid=uuid, include_vector=include_vector
        )
        if result:
            obj_dict = {"uuid": str(result.uuid), "properties": result.properties}
            if include_vector and hasattr(result, "vector") and result.vector:
                obj_dict["vector"] = result.vector
            return {"status": "success", "object": obj_dict}
        else:
            return {"status": "error", "message": "Object not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def update_object(
    class_name: str,
    uuid: str,
    properties: Dict[str, Any],
    vector: Optional[List[float]] = None,
) -> dict:
    """Update an object in a collection."""
    try:
        collection = weaviate_client.client.collections.get(class_name)
        collection.data.update(uuid=uuid, properties=properties, vector=vector)
        return {"status": "success", "message": "Object updated successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def delete_object(class_name: str, uuid: str) -> dict:
    """Delete an object from a collection."""
    try:
        collection = weaviate_client.client.collections.get(class_name)
        collection.data.delete_by_id(uuid=uuid)
        return {"status": "success", "message": "Object deleted successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def query_objects(
    class_name: str,
    limit: int = 10,
    offset: int = 0,
    where_filter: Optional[Dict[str, Any]] = None,
    properties: Optional[List[str]] = None,
    include_vector: bool = False,
) -> dict:
    """Query objects from a collection with filters."""
    try:
        collection = weaviate_client.client.collections.get(class_name)
        result = collection.query.fetch_objects(
            limit=limit, offset=offset, include_vector=include_vector
        )
        objects = []
        for obj in result.objects:
            obj_dict = {"uuid": str(obj.uuid), "properties": obj.properties}
            if include_vector and hasattr(obj, "vector") and obj.vector:
                obj_dict["vector"] = obj.vector
            objects.append(obj_dict)

        return {"status": "success", "objects": objects}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def search_near_text(
    class_name: str,
    query_text: str,
    limit: int = 10,
    distance: Optional[float] = None,
    properties: Optional[List[str]] = None,
    include_vector: bool = False,
) -> dict:
    """Perform a near-text vector search."""
    try:
        collection = weaviate_client.client.collections.get(class_name)
        result = collection.query.near_text(
            query=query_text,
            limit=limit,
            distance=distance,
            include_vector=include_vector,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
        )
        objects = []
        for obj in result.objects:
            obj_dict = {"uuid": str(obj.uuid), "properties": obj.properties}
            if include_vector and hasattr(obj, "vector") and obj.vector:
                obj_dict["vector"] = obj.vector
            if (
                hasattr(obj, "metadata")
                and obj.metadata
                and hasattr(obj.metadata, "distance")
            ):
                obj_dict["distance"] = obj.metadata.distance
            objects.append(obj_dict)

        return {"status": "success", "objects": objects}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def search_near_vector(
    class_name: str,
    vector: List[float],
    limit: int = 10,
    distance: Optional[float] = None,
    properties: Optional[List[str]] = None,
    include_vector: bool = False,
) -> dict:
    """Perform a near-vector search."""
    try:
        collection = weaviate_client.client.collections.get(class_name)
        result = collection.query.near_vector(
            near_vector=vector,
            limit=limit,
            distance=distance,
            include_vector=include_vector,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
        )
        objects = []
        for obj in result.objects:
            obj_dict = {"uuid": str(obj.uuid), "properties": obj.properties}
            if include_vector and hasattr(obj, "vector") and obj.vector:
                obj_dict["vector"] = obj.vector
            if (
                hasattr(obj, "metadata")
                and obj.metadata
                and hasattr(obj.metadata, "distance")
            ):
                obj_dict["distance"] = obj.metadata.distance
            objects.append(obj_dict)

        return {"status": "success", "objects": objects}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def hybrid_search(
    class_name: str,
    query_text: str,
    vector: Optional[List[float]] = None,
    alpha: float = 0.5,
    limit: int = 10,
    properties: Optional[List[str]] = None,
    include_vector: bool = False,
) -> dict:
    """Perform a hybrid search combining keyword and vector search."""
    try:
        collection = weaviate_client.client.collections.get(class_name)
        result = collection.query.hybrid(
            query=query_text,
            vector=vector,
            alpha=alpha,
            limit=limit,
            include_vector=include_vector,
            return_metadata=weaviate.classes.query.MetadataQuery(score=True),
        )
        objects = []
        for obj in result.objects:
            obj_dict = {"uuid": str(obj.uuid), "properties": obj.properties}
            if include_vector and hasattr(obj, "vector") and obj.vector:
                obj_dict["vector"] = obj.vector
            if (
                hasattr(obj, "metadata")
                and obj.metadata
                and hasattr(obj.metadata, "score")
            ):
                obj_dict["score"] = obj.metadata.score
            objects.append(obj_dict)

        return {"status": "success", "objects": objects}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def batch_insert(
    class_name: str, objects: List[Dict[str, Any]], batch_size: int = 100
) -> dict:
    """Insert multiple objects in a batch."""
    try:
        collection = weaviate_client.client.collections.get(class_name)
        with collection.batch.dynamic() as batch:
            for obj in objects:
                batch.add_object(
                    properties=obj.get("properties", obj),
                    uuid=obj.get("uuid"),
                    vector=obj.get("vector"),
                )
        return {
            "status": "success",
            "message": f"Inserted {len(objects)} objects successfully",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def aggregate_objects(
    class_name: str,
    group_by: Optional[str] = None,
    where_filter: Optional[Dict[str, Any]] = None,
    fields: Optional[List[str]] = None,
) -> dict:
    """Aggregate objects in a collection."""
    try:
        collection = weaviate_client.client.collections.get(class_name)
        query_builder = collection.aggregate.over_all(total_count=True)

        if where_filter:
            query_builder = query_builder.with_where(where_filter)

        if group_by:
            query_builder = query_builder.with_group_by([group_by])

        result = query_builder.do()
        return {"status": "success", "aggregation": {"total_count": result.total_count}}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def check_connection() -> dict:
    """Check if Weaviate connection is active."""
    try:
        is_ready = weaviate_client.client.is_ready()
        if is_ready:
            return {
                "status": "success",
                "connected": True,
                "url": weaviate_client.url,
            }
        else:
            return {
                "status": "error",
                "connected": False,
                "message": "Weaviate is not ready",
            }
    except Exception as e:
        return {"status": "error", "connected": False, "message": str(e)}


def get_schema() -> dict:
    """Get the complete Weaviate schema."""
    try:
        collections = weaviate_client.client.collections.list_all()
        schema = {}
        for name, collection in collections.items():
            config = collection.config.get()
            schema[name] = {
                "name": config.name,
                "description": config.description,
                "properties": [
                    {"name": p.name, "data_type": str(p.data_type)}
                    for p in config.properties
                ],
            }
        return {"status": "success", "schema": schema}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def delete_all_objects(
    class_name: str, where_filter: Optional[Dict[str, Any]] = None
) -> dict:
    """Delete all objects from a collection, optionally with a filter."""
    try:
        collection = weaviate_client.client.collections.get(class_name)
        if where_filter:
            collection.data.delete_many(where=where_filter)
        else:
            collection.data.delete_many()
        return {
            "status": "success",
            "message": f"All objects deleted from '{class_name}'",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
