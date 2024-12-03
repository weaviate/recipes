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

    The "Advanced" Weaviate Tool will illustrate a multi-collection query strategy
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

# ToDo -- Switch from Ollama to OpenAI Example

```python
tool_mapping = {
    "get_search_results": get_search_results
}

def ollama_generation_with_tools(user_message: str,
                                 tools_schema: List, tool_mapping: Dict,
                                 model_name: str = "llama3.1") -> str:
    messages=[{
        "role": "user",
        "content": user_message
    }]
    response = ollama.chat(
        model=model_name,
        messages=messages,
        tools=tools_schema
    )
    if not response["message"].get("tool_calls"):
        return response["message"]["content"]
    else:
        for tool in response["message"]["tool_calls"]:
            function_to_call = tool_mapping[tool["function"]["name"]]
            print(f"Calling function {function_to_call}...")
            function_response = function_to_call(tool["function"]["arguments"]["query"])
            messages.append({
                "role": "tool",
                "content": function_response,
            })
    
    final_response = ollama.chat(model=model_name, messages=messages)
    return final_response["message"]["content"]
```

Note, another strategy we use in our Function Calling setups is a `tool_call_budget` that tracks 

# Advanced Weaviate Querying Tool

We will now take a look at a more advanced Weaviate Querying Tool. As the section title suggests, one of the key reasons this is more "advanced" is that we will use all of Weaviate's Query APIs instead of just search queries. This means that the agent can retrieve objects from a Weaviate Collection based on filter matches, rather than just search queries. This is useful for queries such as "Show me my unread emails from last week". Further, this Advanced Weaviate Query Tool will add `collection_name` as an argument, enabling the Agent to send queries across multiple collections stored in Weaviate.

We can further apply Aggregations to the results of a Weaviate query.

...

Groupby and Sort

## 1. Define the Advanced Weaviate Query Tool Schema

## 2. Implement the Tool Execution

## 3. Interface in the Function Calling Loop

This is identical to the Simple Weaviate Query Tool example above