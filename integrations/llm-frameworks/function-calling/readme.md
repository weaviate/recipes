# Function Calling with Weaviate

This guide will explain how to use Weaviate with Function Calling in 3 sections:

1. Introduction to Function Calling
2. Simple Weaviate Search Tool
3. Advanced Weaviate Querying Tool

*\*Please note, `Tool` and `Function` mean the same thing.*

## Introduction to Function Calling

The capabilities of AI systems built with Large Language Models (LLMs) are advancing rapidly largely thanks to their connection with external tools. **Function Calling** has stood the test of time as one of the most powerful design patterns for implementing these connections between LLMs and tools.

Taken from OpenAI.com, "You can define a set of functions as tools that the model has access to, and it can use them when appropriate based on the conversation history. You can then execute those functions on the application side, and provide results back to the model."

This is further visualized in the following image, please note the responsibility of your application to execute the function requested by the model.


Now that you have a sense of how Function Calling works, let's dive further into how we equip LLMs with a set of tool schemas, and loop between LLM responses and tool execution. This is achieved by defining a Tool Schema that describes the name of a Tool and what it does, as well as its respective arguments, and what they do. The LLM then performs inference in a function calling loop, at each step deciding to either complete the response or call one of the external functions.

We will now cover 2 different Weaviate as a Tool setups, a **Simple Weaviate Tool** and an **Advanced Weaviate Tool**. There are three core abstractions underlying all Tool setups:

1. Defining the Tool Schema
2. Implementing the Tool Execution
3. Executing Tools in the Function Calling loop

## Simple Weaviate Search Tool

This Weaviate Tool is defined as Simple because it (1) only searches in a single hard-coded collection and (2) only retrieves data from the collection based on a search query.

## 1. Define a tools_schema such as:

The first thing to note here is that the LLM SDKs expect slightly different Tool Schemas :(.

We have things like LiteLLM and the use of Gemini through OpenAI's SDK that are aiming to unify this.

But for now, we recommend Weaviate users pay attention to this if for example you are switching from Anthropic Function Calling to OpenAI Function Calling.

Here is the OpenAI Tool Schema interfaced as typed Pydantic models:

```python
class OpenAIParameterProperty(BaseModel):
    type: str
    description: str
    enum: Optional[list[str]] = None


class OpenAIParameters(BaseModel):
    type: Literal["object"]
    properties: dict[str, OpenAIParameterProperty]
    required: Optional[list[str]]


class OpenAIFunction(BaseModel):
    name: str
    description: str
    parameters: OpenAIParameters


class OpenAITool(BaseModel):
    type: Literal["function"]
    function: OpenAIFunction
```

Implement `search_blogs` as an OpenAITool:

```python
tools_schema=[
    OpenAITool(
        function=OpenAIFunction(
            name="search_blogs"",
            description="Search for relevant chunks in a collection of Weaviate's Blog Posts.",
            parameters=OpenAIParameters(
                type="object",
                properties=OpenAIParameterProperty(
                    type="str",
                    description="The search query."
                ),
                required=["search_query"]
            )
        )
    )
]
```

## 2. Implement the Tool Execution

```python
def get_search_results(query: str) -> str:
    """Sends a query to Weaviate's Hybrid Search. Parases the response into a {k}:{v} string."""
    
    '''
    Please note, this is a key detail of why we call this a "Simple" Weaviate Tool
    `knowledge_base` is global to the runtime,
    
    So you could simply define this above the function calling block with e.g.
    `knowledge_base = weaviate_client.collections.get("KnowledgeBase")`

    Or what we commonly do is wrap function calling in Classes and save this as internal state with e.g.
    `self.knowledge_base = weaviate_client.collections.get("KnowledgeBase")`

    The "Advanced" Weaviate Tool will illustrate a multi-collection query strategy.
    '''
    response = knowledge_base.query.hybrid(
        query,
        limit=3
    )
    
    stringified_response = ""
    for idx, o in enumerate(response.objects):
        stringified_response += f"\033[92mSearch Result: {idx+1}\033[0m\n"
        for prop in o.properties:
            stringified_response += f"{prop}: {o.properties[prop]}"
        stringified_response += "\n\n"
    
    return stringified_response
```

The key thing to note here is how you scope the connection to Weaviate and/or the collection you are searching in, for example you could also do this:

```python
def search_weaviate_collection(
  weaviate_client: weaviate.WeaviateClient,
  collection_name: str, 
  search_query: str,
  limit: int
  ) -> str:
  """Sends a query to Weaviate’s Hybrid Search."""

  weaviate_client = weaviate.connect_to_wcs(...)

  search_collection = weaviate_client.collections.get(collection_name)
  response = search_collection.query.hybrid(search_query, limit)

  stringified_response = ""

  for idx, o in enumerate(response.objects):
    stringified_response += f"Search Result {idx+1}:\n"
    for prop in o.properties:
        stringified_response+=f"{prop}:{o.properties[prop]}"
    stringified_response += "\n"

    weaviate_client.close()

  return stringified_response
```

Most LLM SDKs that offer Function Calling also offer **Parallel** Function Calling. We can further speed this up by implementing our Weaviate Searches to run asynchronously so that each search request can run in parallel. Note, to achieve this make the following change in your search function.

```python
async def search_weaviate_collection(
    weaviate_client: weaviate.WeaviateAsyncClient,
    # ... everything else remains the same
):
    # .... everything else remains the same
    response = await search_collection.query.hybrid(search_query, limit)
    # ... everything else remains the same
```

There is a little more to how this changes the typical function calling loop to enable parallel function calling as well which we will see below, starting without the use of parallel function calling and async weaviate querying.

## 3. Extend LLM generation with the function calling **loop**



```python
tool_mapping = {
    "get_search_results": get_search_results
}

def openai_function_calling_loop(user_message: str, openai_client: openai.OpenAI,
                                 tools: List[OpenAITool], tool_mapping: Dict,
                                 call_budget: int = 5,
                                 model_name: str = "gpt-4o") -> str:
    messages=[{
        "role": "user",
        "content": user_message
    }]
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        parallel_tool_calls=True
    )

    while calls < call_budget:
        message = response.choices[0].message
    
        if not message.tool_calls
            return message.content

        else:
            for tool_call in message.tool_calls:
                function_to_call = tool_call.function.name
                print(f"Calling function {function_to_call}...")
                tool_arguments = json.loads(tool_call.function.arguments)
                function_response = function_to_call(tool_arguments),
                messages.append({
                    "role": "tool",
                    "content": function_response,
                }) 
```

### Async Tool Execution

# Advanced Weaviate Querying Tool

We will now take a look at a more advanced Weaviate Querying Tool. As the section title suggests, one of the key reasons this is more "advanced" is that we will use all of Weaviate's Query APIs instead of just search queries. This means that the agent can retrieve objects from a Weaviate Collection based on filter matches, rather than just search queries. This is useful for queries such as "Show me my unread emails from last week". Further, this Advanced Weaviate Query Tool will add `collection_name` as an argument, enabling the Agent to send queries across multiple collections stored in Weaviate.

We can further apply Aggregations to the results of a Weaviate query.

...

Groupby and Sort

## 1. Define the Advanced Weaviate Query Tool Schema

```python
def get_weaviate_gorilla_tool(
    collections_description: str, collections_enum: list[str]
) -> Tool:
    properties: Dict[str, Dict[str, Any]] = {
            "collection_name": {
                "type": "string",
                "description": "The collection to query.",
                "enum": collections_enum
            },
            "search_query": {
                "type": "string",
                "description": "A search query to return objects from a search index."
            },
            "integer_property_filter": {
                "type": "object",
                "description": "Filter numeric properties using comparison operators.",
                "properties": {
                    "property_name": {"type": "string"},
                    "operator": {"type": "string", "enum": ["=", "<", ">", "<=", ">="]},
                    "value": {"type": "number"}
                },
            },
            "text_property_filter": {
                "type": "object", 
                "description": "Filter text properties using equality or LIKE operators",
                "properties": {
                    "property_name": {"type": "string"},
                    "operator": {"type": "string", "enum": ["=", "LIKE"]},
                    "value": {"type": "string"}
                }
            },
            "boolean_property_filter": {
                "type": "object",
                "description": "Filter boolean properties using equality operators",
                "properties": {
                    "property_name": {"type": "string"},
                    "operator": {"type": "string", "enum": ["=", "!="]},
                    "value": {"type": "boolean"}
                }
            },
            "integer_property_aggregation": {
                "type": "object",
                "description": "Aggregate numeric properties using statistical functions",
                "properties": {
                    "property_name": {"type": "string"},
                    "metrics": {"type": "string", "enum": ["COUNT", "TYPE", "MIN", "MAX", "MEAN", "MEDIAN", "MODE", "SUM"]}
                }
            },
            "text_property_aggregation": {
                "type": "object",
                "description": "Aggregate text properties using frequency analysis",
                "properties": {
                    "property_name": {"type": "string"},
                    "metrics": {"type": "string", "enum": ["COUNT", "TYPE", "TOP_OCCURRENCES"]},
                    "top_occurrences_limit": {"type": "integer"}
                }
            },
            "boolean_property_aggregation": {
                "type": "object",
                "description": "Aggregate boolean properties using statistical functions",
                "properties": {
                    "property_name": {"type": "string"},
                    "metrics": {"type": "string", "enum": ["COUNT", "TYPE", "TOTAL_TRUE", "TOTAL_FALSE", "PERCENTAGE_TRUE", "PERCENTAGE_FALSE"]}
                }
            },
            "groupby_property": {
                "type": "string",
                "description": "Group the results by a property."
            }
        }
    return Tool(
        type="function",
        function=Function(
            name="query_database",
            description=f"""Query a database with an optional search query or optional filters or aggregations on the results.

            IMPORTANT! Please be mindful of the available query APIs you can use such as search queries, filters, aggregations, and groupby!

            Available collections in this database:
            {collections_description}""",
            parameters=Parameters(
                type="object",
                properties=properties,
                required=["collection_name"]
            )
        )
    )

collections_description = "\n".join(
    [c.description for c in self.collections if c.description]
)

collections_enum = [c.name for c in self.collections]

tools = [get_weaviate_gorilla_tool(
    collections_description=collections_description,
    collections_enum=collections_enum
)]
```

## 2. Implement the Tool Execution

```python

```

## 3. Interface in the Function Calling Loop

This is identical to the Simple Weaviate Query Tool example above