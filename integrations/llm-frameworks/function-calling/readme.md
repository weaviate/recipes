# Function Calling with Weaviate

## Quick Reference:

```python
def code_search(query: str) -> str:
    code_collection = client.collections.get("Code")
    response = code_collection.query.hybrid(query, limit=5)

    stringified_response = ""
    for idx, o in enumerate(response.objects):
        stringified_response += f"Search Result: {idx+1}:\n"
        for prop in o.properties:
            stringified_response += f"{prop}: {o.properties[prop]}\n"
        stringified_response += "\n"

    return stringified_response
```

## Detailed Guide to Function Calling with Weaviate

Function calling is one of the most powerful emerging design patterns in AI-native applications. Function calling describes equipping Foundation Models, such as Large Language Models (LLMs), with external tools. More particularly, this is achieved by defining a schema that describes the name of a function and what it does, as well as its respective arguments, and what they do. The LLM then performs inference in a function calling loop, at each step deciding to either complete the response or call one of the external functions.

There are three core abstractions underlying most function calling frameworks:

## 1. Define a tools_schema such as:

```python
tools_schema=[{
    'type': 'function',
    'function': {
        'name': 'search_weaviate_collection',
        'description': 'Get search results for a provided query from a provided collection of data.',
        'parameters': {
          'type': 'object',
          'properties': {
            'collection_name': {
              'type': 'string',
              'description': 'The collection in Weaviate to search through.',
            },
            'query': {
              'type': 'string',
              'description': 'The search query.',
            },
          },
          'required': ['query'],
        },
    },
}]
```

## 2. Define a mapping from the names of functions to where they will be executed (e.g. in the Python runtime)

```python
tool_mapping = {
    "get_search_results": get_search_results
}
```

Note, `get_search_results` could wrap a request to an API not defined in the same runtime as the function calling loop.

## 3. Extend LLM generation with the function calling **loop**

```python
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

# Add Weaviate Search to Function Calling

Building an external search function is pretty straightforward. We wrap our Weaviate hybrid search in a function that takes a search_query string as an argument and the collection_name string as the argument of which Weaviate collection to search through.

```python
def search_weaviate_collection(
  weaviate_client: weaviate.WeaviateClient,
  collection_name: str, 
  search_query: str,
  limit: int
  ) -> str:
  """Sends a query to Weaviate’s Hybrid Search."""

  search_collection = weaviate_client.collections.get(collection_name)
  response = search_collection.query.hybrid(search_query, limit)

  stringified_response = ""

  for idx, o in enumerate(response.objects):
    stringified_response += f"Search Result {idx+1}:\n"
    for prop in o.properties:
        stringified_response+=f"{prop}:{o.properties[prop]}"
    stringified_response += "\n"

  return stringified_response
```

# Add Weaviate Filters to Function Calling

What if we want to add Weaviate's filters to our function calling systems? This requires taking the training wheels off a bit since Weaviate's filter syntax is a more complex argument type.

## Filter Parser

```python
from weaviate.classes.query import Filter

def build_weaviate_filter(filter_string: str) -> Filter:
    def parse_condition(condition: str) -> Filter:
        parts = condition.split(':')
        if len(parts) < 3:
            raise ValueError(f"Invalid condition: {condition}")
        
        property, operator, value = parts[0], parts[1], ':'.join(parts[2:])
        
        if operator == '==':
            return Filter.by_property(property).equal(value)
        elif operator == '!=':
            return Filter.by_property(property).not_equal(value)
        elif operator == '>':
            return Filter.by_property(property).greater_than(float(value))
        elif operator == '<':
            return Filter.by_property(property).less_than(float(value))
        elif operator == '>=':
            return Filter.by_property(property).greater_than_equal(float(value))
        elif operator == '<=':
            return Filter.by_property(property).less_than_equal(float(value))
        elif operator == 'LIKE':
            return Filter.by_property(property).like(value)
        elif operator == 'CONTAINS_ANY':
            return Filter.by_property(property).contains_any(value.split(','))
        elif operator == 'CONTAINS_ALL':
            return Filter.by_property(property).contains_all(value.split(','))
        elif operator == 'WITHIN':
            lat, lon, dist = map(float, value.split(','))
            return Filter.by_property(property).within_geo_range(lat, lon, dist)
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    def parse_group(group: str) -> Filter:
        if 'AND' in group:
            conditions = [parse_group(g.strip()) for g in group.split('AND')]
            return Filter.all_of(conditions)
        elif 'OR' in group:
            conditions = [parse_group(g.strip()) for g in group.split('OR')]
            return Filter.any_of(conditions)
        else:
            return parse_condition(group)

    # Remove outer parentheses if present
    filter_string = filter_string.strip()
    if filter_string.startswith('(') and filter_string.endswith(')'):
        filter_string = filter_string[1:-1]

    return parse_group(filter_string)
```

Example:

```python
filter_string = "category:==:Python AND (points:>:300 OR difficulty:LIKE:*hard*)"
parsed_filter = build_weaviate_filter(filter_string)
print(parsed_filter)

# <weaviate.collections.classes.filters._FilterAnd object at 0x106109570>
```

## Filtering with `get_objects_with_filters` and `search_collection_with_filters`

```python
def get_objects_with_filters(
  weaviate_client: weaviate.WeaviateClient,
  collection_name: str,
  filters: str,
  limit: int
) -> str:
  """Sends a query to Weaviate's /objects API."""

  search_collection = weaviate_client.collection.get(collection_name)
  filters = build_weaviate_filter(filters)
  response = search_collection.query.fetch_objects(
    filters=filters,
    limit=limit
  )

  stringified_response = ""

  for idx, o in enumerate(response.objects):
    stringified_response += f"Search Result {idx+1}:\n"
    for prop in o.properties:
        stringified_response+=f"{prop}:{o.properties[prop]}"
    stringified_response += "\n"

  return stringified_response
```

### Let's also add, `search_collection_with_filters`

```python
from typing import Optional

def search_collection_with_filters(
  weaviate_client: weaviate.WeaviateClient,
  collection_name: str,
  search_query: str,
  limit: int,
  filter_string: Optional[str] = None,
) -> str:
  """Sends a query to Weaviate's /objects API."""
  search_collection = weaviate_client.collection.get(collection_name)

  if filters:
    filters = build_weaviate_filter(filter_string)

    response = search_collection.query.hybrid(
       query=search_query,
       filters=filters,
       limit=limit  
    )
  else:
     response = search_collection.query.hybrid(
        query=search_query,
        limit=limit
     )

  stringified_response = ""

  for idx, o in enumerate(response.objects):
      stringified_response += f"Search Result {idx+1}:\n"
      for prop in o.properties:
          stringified_response+=f"{prop}:{o.properties[prop]}"
      stringified_response += "\n"

  return stringified_response
```

# 3. Symbolic Aggregation

Weaviate also supports symbolic aggregrations, such as calculating the average or maximum integer-valued age property. Weaviate also supports grouping objects by values and further computing such aggregations, for example we can group product items by category, such as "table", "chair", or "lamp", and then compute the average age of each category.

To enable Weaviate's Symbolic Aggregations in Function Calling, we will again construct a Domain-Specific Language for it, similar to Filtering.
