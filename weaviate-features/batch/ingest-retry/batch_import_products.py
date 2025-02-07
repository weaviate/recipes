import weaviate
import weaviate.classes.config as wc
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.util import generate_uuid5
import os
import json
import ijson
import sys
import argparse

WEAVIATE_URL = "http://localhost:8080"

OLLAMA_EMBEDDING_MODEL_ID = "mxbai-embed-large:latest"
OLLAMA_GENERATIVE_MODEL_ID = "llama3.2:latest"
OLLAMA_URL = "http://host.docker.internal:11434"

PRODUCT_COLLECTION_NAME = "product"

def validate_input():
    parser = argparse.ArgumentParser(description="Import products to Weaviate.")
    parser.add_argument("json_file", help="Path to the JSON file containing product data.")
    parser.add_argument("--drop", action="store_true", help="Drop existing collection before import.")
    args = parser.parse_args()
    
    if not os.path.isfile(args.json_file):
        print(f"Error: The file '{args.json_file}' does not exist.")
        sys.exit(1)
    if not args.json_file.endswith('.json'):
        print(f"Error: The file '{args.json_file}' does not have a .json extension.")
        sys.exit(1)
    
    return args.json_file, args.drop

def connect_to_weaviate():
    headers = {}
    client = weaviate.connect_to_local(
        headers=headers,     
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=60, insert=120)
        )
    )
    assert client.is_live()
    return client

def drop_product_collection(client):
    if client.collections.exists(PRODUCT_COLLECTION_NAME):
        print("Dropping existing product collection")
        client.collections.delete(PRODUCT_COLLECTION_NAME)

def create_product_collection(client):
    if not client.collections.exists(PRODUCT_COLLECTION_NAME):
        print("Creating product collection")
    
    print("Creating product collection")
    client.collections.create(
        name=PRODUCT_COLLECTION_NAME,
        properties=[
            wc.Property(name="category", data_type=wc.DataType.TEXT_ARRAY, index_filterable=True, index_searchable=True),
            wc.Property(name="tech1", data_type=wc.DataType.TEXT, skip_vectorization=True, index_filterable=False, index_searchable=False),
            wc.Property(name="tech2", data_type=wc.DataType.TEXT, skip_vectorization=True, index_filterable=False, index_searchable=False),
            wc.Property(name="description", data_type=wc.DataType.TEXT_ARRAY, index_filterable=True, index_searchable=True),
            wc.Property(name="fit", data_type=wc.DataType.TEXT, skip_vectorization=True, index_filterable=False, index_searchable=False),
            wc.Property(name="title", data_type=wc.DataType.TEXT, index_filterable=True, index_searchable=True),
            wc.Property(name="also_buy", data_type=wc.DataType.TEXT_ARRAY, skip_vectorization=True, index_filterable=False, index_searchable=False),
            wc.Property(name="image", data_type=wc.DataType.TEXT_ARRAY, skip_vectorization=True, index_filterable=False, index_searchable=False),
            wc.Property(name="brand", data_type=wc.DataType.TEXT, index_filterable=True, index_searchable=True),
            wc.Property(name="feature", data_type=wc.DataType.TEXT_ARRAY, skip_vectorization=True, index_filterable=False, index_searchable=False),
            wc.Property(name="rank", data_type=wc.DataType.TEXT_ARRAY, skip_vectorization=True, index_filterable=False, index_searchable=False),
            wc.Property(name="also_view", data_type=wc.DataType.TEXT_ARRAY, skip_vectorization=True, index_filterable=False, index_searchable=False),
            wc.Property(name="main_cat", data_type=wc.DataType.TEXT, index_filterable=True, index_searchable=True),
            wc.Property(name="date", data_type=wc.DataType.TEXT, skip_vectorization=True, index_filterable=True, index_searchable=True),
            wc.Property(name="price", data_type=wc.DataType.TEXT, skip_vectorization=True, index_filterable=True, index_searchable=True),
            wc.Property(name="asin", data_type=wc.DataType.TEXT, index_filterable=True, index_searchable=True),
        ],
        vectorizer_config=wc.Configure.Vectorizer.text2vec_ollama(
            api_endpoint=OLLAMA_URL,
            model=OLLAMA_EMBEDDING_MODEL_ID,
        ),
        generative_config=wc.Configure.Generative.ollama(
            api_endpoint=OLLAMA_URL,
            model=OLLAMA_GENERATIVE_MODEL_ID
        )
    )
    print("Product collection created")

def normalize_field(obj, field):
    if not isinstance(obj[field], list):
        obj[field] = [obj[field]]

def process_product(obj):
    product_obj = {
        "category": obj["category"],
        "tech1": obj["tech1"],
        "tech2": obj["tech2"],
        "description": obj["description"],
        "fit": obj["fit"],
        "title": obj["title"],
        "also_buy": obj["also_buy"],
        "image": obj["image"],
        "brand": obj["brand"],
        "feature": obj["feature"],
        "rank": obj["rank"],
        "also_view": obj["also_view"],
        "main_cat": obj["main_cat"],
        "date": obj["date"],
        "price": obj["price"],
        "asin": obj["asin"],
    }
    
    for field in ["category", "description", "also_buy", "image", "feature", "also_view"]:
        normalize_field(product_obj, field)
    
    # Sometimes rank is a string and sometimes it is an array
    if isinstance(obj["rank"], str):
        product_obj["rank"] = [obj["rank"]]
    elif isinstance(obj["rank"], list):
        product_obj["rank"] = obj["rank"]
    else:
        product_obj["rank"] = []
    
    return product_obj

def import_products(products, local_json_path):
    counter = 0
    INTERVAL = 100

    with products.batch.dynamic() as batch:
        print(f"Opening {local_json_path}")
        with open(local_json_path, "rb") as f:
            objects = ijson.items(f, '', multiple_values=True)
            for obj in objects:
                product_obj = process_product(obj)
                #print(json.dumps(product_obj, indent=2))
                batch.add_object(
                    properties=product_obj,
                    uuid=generate_uuid5(obj["asin"])
                )

                counter += 1
                if counter % INTERVAL == 0:
                    print(f"{local_json_path}: Imported {counter} products...")
        print(f"{local_json_path}: Flushing batch")
        batch.flush()
        print(f"{local_json_path}: Batch flushed")

    # The failed_objects are not available until after flush is called
    old_failed_obj_count = len(products.batch.failed_objects)
    new_failed_obj_count = 0
    while True:
        if len(products.batch.failed_objects) == 0:
            print(f"{local_json_path}: All products imported successfully")
            break
        
        print(f"{local_json_path}: Retrying {len(products.batch.failed_objects)} failed objects...")
        retry_counter = 0

        current_failed_object_count = len(products.batch.failed_objects)
        failed_objects = products.batch.failed_objects
        with products.batch.dynamic() as batch:
            print(f"{local_json_path}: Inside retry loop are {len(failed_objects)} failed objects...")

            for failed in failed_objects:
                try:
                    print(f"{local_json_path}: Failed with error \"{failed.message}\": {failed.object_.uuid}")
                    #print(f"{local_json_path}: " 
                    #    + json.dumps(failed.object_.properties, indent=2))
                    if new_failed_obj_count == old_failed_obj_count:
                        print(f"{local_json_path}: Debugging stuck object: " 
                                + json.dumps(failed.object_.properties, indent=2))
                    batch.add_object(
                        properties=failed.object_.properties,
                        uuid=failed.object_.uuid
                    )
                except Exception as e:
                    print(f"{local_json_path}: Exception while retrying: {e}")
                    print(f"{local_json_path}: Failed Object: {failed}")
                    break

                retry_counter += 1
                if retry_counter % INTERVAL == 0:
                    print(f"{local_json_path}: Retried {retry_counter} products...")
            batch.flush()    
        old_failed_obj_count = current_failed_object_count
        new_failed_obj_count = len(products.batch.failed_objects)

def main():
    local_json_path, drop_collection = validate_input()
    client = None

    try:
        client = connect_to_weaviate()
        #metainfo = client.get_meta()
        #print(json.dumps(metainfo, indent=2))  # Print the meta information in a readable format
        if drop_collection:
            drop_product_collection(client)
            create_product_collection(client)
        elif not client.collections.exists(PRODUCT_COLLECTION_NAME):
            create_product_collection(client)
        products = client.collections.get(PRODUCT_COLLECTION_NAME)
        aggregation = products.aggregate.over_all(total_count=True)
        print(f"{local_json_path}: Original total count = {aggregation.total_count}")
        import_products(products, local_json_path)
        aggregation = products.aggregate.over_all(total_count=True)
        print(f"{local_json_path}: New total count = {aggregation.total_count}")
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        if client:
            client.close()

if __name__ == "__main__":
    main()
