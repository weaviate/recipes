# Weaviate ADK Agent

A comprehensive Weaviate vector database agent built with Google's Agent Development Kit (ADK). This agent provides natural language interaction with Weaviate, enabling full CRUD operations, vector search, collection management, and more through an intelligent conversational interface.

## Features

- **Collection Management**: Create, delete, list collections, and manage schemas
- **Data Operations**: Insert, retrieve, update, and delete objects
- **Vector Search**: Hybrid search combining keyword and vector search for optimal results
- **Batch Operations**: Efficiently insert multiple objects
- **Aggregations**: Analyze and aggregate data across collections
- **Connection Management**: Check connection status and reconnect as needed
- **Natural Language Interface**: Interact with Weaviate using conversational commands powered by Gemini 3 Pro

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up environment variables:

Create a `.env` file in the project root with the following required variables:

```env
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
GOOGLE_CLOUD_PROJECT=your-project-id
```

**Required Environment Variables:**

- `WEAVIATE_URL` - The URL of your Weaviate instance (defaults to `http://localhost:8080` if not set)
- `WEAVIATE_API_KEY` - API key for Weaviate authentication (required for Weaviate Cloud, optional for local instances)
- `GOOGLE_API_KEY` - Google API key for Gemini API access (required for ADK agent)
- `GOOGLE_CLOUD_PROJECT` - Your Google Cloud project ID (required for ADK agent)

## Configuration

The agent supports multiple Weaviate deployment types:

- **Local Weaviate**: Automatically detected when URL contains `localhost`
- **Weaviate Cloud**: Automatically detected when URL contains `weaviate.io` or `weaviate.cloud` (requires API key)
- **Custom Weaviate**: Any other Weaviate instance (supports both HTTP and HTTPS)

## Usage

### Basic Usage

```python
from weaviate_agent import root_agent

# The agent is ready to use
# You can interact with it through the ADK framework
```

### Example Interactions

The agent understands natural language commands. Here are some example interactions:

- "Check if Weaviate is connected"
- "Create a collection called 'Articles' with a text property"
- "Insert an article with title 'AI Overview' and content 'Artificial intelligence...'"
- "Search for articles about machine learning"
- "List all collections"
- "Get the schema for the Articles collection"
- "Delete all objects from the Articles collection"

## Available Tools

The agent has access to the following tools:

### Connection Management

- `check_connection()` - Verify Weaviate connection status

### Collection Management

- `create_collection(class_name, description, properties, vectorizer)` - Create a new collection
- `delete_collection(class_name)` - Delete a collection
- `list_collections()` - List all collections
- `get_collection_schema(class_name)` - Get schema for a specific collection
- `get_schema()` - Get complete Weaviate schema

### Data Operations

- `insert_object(class_name, properties, uuid, vector)` - Insert a single object
- `get_object(class_name, uuid, include_vector)` - Retrieve an object by UUID
- `update_object(class_name, uuid, properties, vector)` - Update an object
- `delete_object(class_name, uuid)` - Delete an object
- `delete_all_objects(class_name, where_filter)` - Delete all objects (with optional filter)

### Search Operations

- `query_objects(class_name, limit, offset, where_filter, properties, include_vector)` - Query objects with filters
- `hybrid_search(class_name, query_text, vector, alpha, limit, properties, include_vector)` - Hybrid search (keyword + vector)
- `search_near_text(class_name, query_text, limit, distance, properties, include_vector)` - Vector search by text
- `search_near_vector(class_name, vector, limit, distance, properties, include_vector)` - Vector search by vector

### Batch Operations

- `batch_insert(class_name, objects, batch_size)` - Insert multiple objects efficiently

### Aggregations

- `aggregate_objects(class_name, group_by, where_filter, fields)` - Aggregate data from collections

## Requirements

- Python 3.8+
- `weaviate-client>=4.0.0`
- `python-dotenv>=1.0.0`
- Google ADK (Agent Development Kit)
- Access to Gemini 3 Pro Preview model

## Architecture

The agent consists of two main components:

1. **`agent.py`**: Defines the root agent with instructions and tool configuration
2. **`weaviate_tools.py`**: Contains all Weaviate interaction functions and the WeaviateClient class

The agent uses the Gemini 3 Pro Preview model to understand natural language requests and automatically select the appropriate tools to fulfill them.

## Notes

- The agent automatically checks connection status when starting operations
- Hybrid search is recommended for all search operations as it combines keyword and vector search
- The agent handles both local and cloud Weaviate deployments automatically
- All operations return structured responses with status and relevant data
