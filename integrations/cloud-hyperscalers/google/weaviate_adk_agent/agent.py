from google.adk.agents.llm_agent import Agent
from .weaviate_tools import (
    check_connection,
    create_collection,
    delete_collection,
    list_collections,
    get_collection_schema,
    insert_object,
    get_object,
    update_object,
    delete_object,
    query_objects,
    hybrid_search,
    batch_insert,
    aggregate_objects,
    get_schema,
    delete_all_objects,
    weaviate_client,
)


root_agent = Agent(
    model="gemini-3-pro-preview",
    name="weaviate_agent",
    description="A comprehensive Weaviate vector database agent with full CRUD, search, and management capabilities.",
    instruction="""You are a Weaviate vector database assistant. You have access to all Weaviate features including:
- Collection management: create, delete, list collections, and manage schemas
- Data operations: insert, get, update, delete objects
- Vector search: hybrid search (combining keyword and vector search)
- Batch operations: insert multiple objects efficiently
- Aggregations: analyze and aggregate data
- Connection management: check connection status and reconnect

Always check the connection first when starting. Use appropriate tools based on user requests.
For all search operations, use hybrid_search which combines keyword and vector search for best results.
Return clear, structured responses with status and relevant data.""",
    tools=[
        check_connection,
        create_collection,
        delete_collection,
        list_collections,
        get_collection_schema,
        insert_object,
        get_object,
        update_object,
        delete_object,
        query_objects,
        hybrid_search,
        batch_insert,
        aggregate_objects,
        get_schema,
        delete_all_objects,
    ],
)
