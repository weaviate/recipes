---
layout: recipe
toc: True
title: "Weaviate Query Agent with Haystack"
featured: False
integration: True
agent: False
tags: ['Query Agent', 'Integration']
---
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weaviate/recipes/blob/main/integrations/llm-agent-frameworks/haystack/haystack-query-agent-tool.ipynb)

## Weaviate Query Agent with Haystack

This notebook will show you how to define the Weaviate Query Agent as a tool through Haystack.

### Requirements
1. Weaviate Cloud instance (WCD): The Weaviate Query Agent is only accessible through WCD at the moment. You can create a serverless cluster or a free 14-day sandbox [here](https://console.weaviate.cloud/).
1. Install Haystack with `pip install haystack-ai`
1. Install the Weaviate Agents package with `pip install weaviate-agents`
1. You'll need a Weaviate cluster with data. If you don't have one, check out [this notebook](https://github.com/weaviate/recipes/blob/main/integrations/Weaviate-Import-Example.ipynb) to import the Weaviate Blogs.

### Import libraries and keys

```python
import weaviate
from weaviate_agents.query import QueryAgent
import os
import json

from haystack.tools import Tool
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
```

```python
os.environ["WEAVIATE_URL"] = ""
os.environ["WEAVIATE_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
```

### Define Query Agent function

```python
def send_query_agent_request(query: str) -> str:
    """
    Send a query to the database and get the response.

    Args:
        query (str): The question or query to search for in the database. This can be any natural language question related to the content stored in the database.

    Returns:
        str: The response from the database containing relevant information.
    """

    # connect to your Weaviate Cloud instance
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}, # add the API key to the model provider from your Weaviate collection
    )

    # connect the query agent to your Weaviate collection(s)
    query_agent = QueryAgent(
        client=weaviate_client,
        collections=["Blogs"]
    )
    return query_agent.run(query).final_answer
```

### Define tool

```python
parameters = {
    "type": "object",
    "properties": {
        "query": {"type": "string"}
    },
    "required": ["query"]
}

query_agent_tool = Tool(
    name="weaviate_query_agent_tool",
    description="This tool queries a database containing blog content about Weaviate and returns relevant information. You can ask any natural language question about the blogs stored in the database.",
    parameters=parameters,
    function=send_query_agent_request
)

# Example usage:
print(query_agent_tool.tool_spec)
print(query_agent_tool.invoke(query="What are the main topics covered in the blogs?"))
```

Python output:
```text
{'name': 'weaviate_query_agent_tool', 'description': 'This tool queries a database containing blog content about Weaviate and returns relevant information. You can ask any natural language question about the blogs stored in the database.', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}
The main topic covered in the blogs is Docker and Containers, with a focus on their use with Weaviate. The articles provide background information on Docker and containers, explain their importance for Weaviate users, and cover topics such as Docker installation and setup, isolation and predictability of environments, distribution via Docker Hub, and the use of Docker Compose. Additionally, they address questions about Docker's role in the Weaviate stack, outlining reasons such as portability, isolation, and dependency management. There’s also discussion on deploying Weaviate using Kubernetes and Helm for more stable environments.

/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/weaviate/warnings.py:314: ResourceWarning: Con004: The connection to Weaviate was not closed properly. This can lead to memory leaks.
            Please make sure to close the connection using `client.close()`.
  warnings.warn(
```
### Chat Conversation with Tool Invocation

```python
chat_generator = OpenAIChatGenerator(model="gpt-4o-mini", tools=[query_agent_tool])
tool_invoker = ToolInvoker(tools=[query_agent_tool])

user_message = ChatMessage.from_user("How do I run Weaviate with Docker?")

replies = chat_generator.run(messages=[user_message])["replies"]
print(f"assistant messages: {replies}")

if replies[0].tool_calls:
    tool_messages = tool_invoker.run(messages=replies)["tool_messages"]
    print(f"tool messages: {tool_messages}")
    # we pass all the messages to the Chat Generator
    messages = [user_message] + replies + tool_messages
    final_replies = chat_generator.run(messages=messages)["replies"]
    print(f"final assistant messages: {final_replies}")
```

Python output:
```text
assistant messages: [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[ToolCall(tool_name='weaviate_query_agent_tool', arguments={'query': 'How to run Weaviate with Docker?'}, id='call_fpBdeb9qHLiifdfFZ5OnmPoE')], _name=None, _meta={'model': 'gpt-4o-mini-2024-07-18', 'index': 0, 'finish_reason': 'tool_calls', 'usage': {'completion_tokens': 27, 'prompt_tokens': 83, 'total_tokens': 110, 'completion_tokens_details': CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), 'prompt_tokens_details': PromptTokensDetails(audio_tokens=0, cached_tokens=0)}})]

/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/weaviate/warnings.py:314: ResourceWarning: Con004: The connection to Weaviate was not closed properly. This can lead to memory leaks.
            Please make sure to close the connection using `client.close()`.
  warnings.warn(

tool messages: [ChatMessage(_role=<ChatRole.TOOL: 'tool'>, _content=[ToolCallResult(result='To run Weaviate with Docker, you need to follow these steps:\n\n1. **Install Docker and Docker Compose**: Make sure you have both Docker and Docker Compose installed on your computer. Installation instructions vary depending on your operating system:\n   - [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)\n   - [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)\n   - [Docker for Ubuntu Linux](https://docs.docker.com/engine/install/ubuntu/) for Docker installation, and [Docker Compose for Ubuntu Linux](https://docs.docker.com/compose/install) for Compose installation.\n\n2. **Obtain the Docker Compose File**: You can obtain a `docker-compose.yml` file directly from the Weaviate documentation. This file defines the services that will be created for Weaviate.\n   - Use the Weaviate configuration tool available on their website to customize and download this file according to your needs.\n\n3. **Run Docker Compose**:\n   - Ensure you are in the directory where the `docker-compose.yml` file is located.\n   - Run the command `docker compose up -d`. The `-d` flag means "detach", so your terminal will not attach to the logs and you can continue using it for other commands.\n\n4. **Check Weaviate\'s Readiness**:\n   - Weaviate has a readiness check endpoint which can be accessed at `GET /v1/.well-known/ready` on the service address. You can use a command like `curl` to check if Weaviate is up, e.g., `curl --fail -s localhost:8080/v1/.well-known/ready`. The service is ready if you receive a `2xx` HTTP status code.\n\nThis setup is suitable for local development and testing. For production environments, running Weaviate on Kubernetes is recommended.', origin=ToolCall(tool_name='weaviate_query_agent_tool', arguments={'query': 'How to run Weaviate with Docker?'}, id='call_fpBdeb9qHLiifdfFZ5OnmPoE'), error=False)], _name=None, _meta={})]
final assistant messages: [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text="To run Weaviate with Docker, follow these steps:\n\n1. **Install Docker and Docker Compose**:\n   - Make sure you have both Docker and Docker Compose installed on your computer. Installation instructions can be found on their respective websites:\n      - [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)\n      - [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)\n      - [Docker for Ubuntu Linux](https://docs.docker.com/engine/install/ubuntu/) for Docker and [Docker Compose for Ubuntu Linux](https://docs.docker.com/compose/install) for Compose.\n\n2. **Obtain the Docker Compose File**:\n   - Get a `docker-compose.yml` file from Weaviate's documentation or use their configuration tool to customize and download it as per your requirements.\n\n3. **Run Docker Compose**:\n   - Navigate to the directory containing the `docker-compose.yml` file.\n   - Execute the command `docker compose up -d`. The `-d` flag runs it in detached mode, allowing you to continue using the terminal.\n\n4. **Check Weaviate's Readiness**:\n   - Use the readiness check endpoint available at `GET /v1/.well-known/ready`. You can use a command like `curl` to verify if Weaviate is ready: `curl --fail -s localhost:8080/v1/.well-known/ready`. A `2xx` HTTP status code indicates that the service is ready.\n\nThis setup is ideal for local development and testing; for production, consider deploying Weaviate on Kubernetes.")], _name=None, _meta={'model': 'gpt-4o-mini-2024-07-18', 'index': 0, 'finish_reason': 'stop', 'usage': {'completion_tokens': 334, 'prompt_tokens': 511, 'total_tokens': 845, 'completion_tokens_details': CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), 'prompt_tokens_details': PromptTokensDetails(audio_tokens=0, cached_tokens=0)}})]
```