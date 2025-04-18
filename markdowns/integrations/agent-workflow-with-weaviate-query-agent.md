---
layout: recipe
colab: https://colab.research.google.com/github/weaviate/recipes/blob/main/integrations/llm-agent-frameworks/llamaindex/agents/agent-workflow-with-weaviate-query-agent.ipynb
toc: True
title: "Weaviate Query Agent with LlamaIndex"
featured: False
integration: True
agent: False
tags: ['Query Agent', 'Integration']
---
<a href="https://colab.research.google.com/github/weaviate/recipes/blob/main/integrations/llm-agent-frameworks/llamaindex/agents/agent-workflow-with-weaviate-query-agent.ipynb" target="_blank">
  <img src="https://img.shields.io/badge/Open%20in-Colab-4285F4?style=flat&logo=googlecolab&logoColor=white" alt="Open In Google Colab" width="130"/>
</a>

## Weaviate Query Agent with LlamaIndex

This notebook will show you how to define the Weaviate Query Agent as a tool through LlamaIndex.

### Requirements
1. Weaviate Cloud instance (WCD): The Weaviate Query Agent is only accessible through WCD at the moment. You can create a serverless cluster or a free 14-day sandbox [here](https://console.weaviate.cloud/).
2. Install LlamaIndex with `pip install llama-index` (we used version `0.12.22` for this notebook)
3. Install the Weaviate Agents package with `pip install weaviate-agents`
4. You'll need a Weaviate cluster with data. If you don't have one, check out [this notebook](https://github.com/weaviate/recipes/blob/main/integrations/Weaviate-Import-Example.ipynbb) to import the Weaviate Blogs.

### Resources on the LlamaIndex Agent Workflow
1. [Getting Started Guide](https://docs.llamaindex.ai/en/latest/getting_started/starter_example/)
1. [Agent Tutorial](https://docs.llamaindex.ai/en/latest/understanding/agent/)
1. [Key Features in the Agent Workflow](https://docs.llamaindex.ai/en/latest/examples/agent/agent_workflow_basic/)

### Import libraries and keys

```python
import weaviate
from weaviate_agents.query import QueryAgent
import os
import json

from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import AgentWorkflow
```

```python
os.environ["WEAVIATE_URL"] = ""
os.environ["WEAVIATE_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
```

### Define Query Agent function

```python
def query_agent_request(query: str) -> str:
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
        headers={ "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY") 
        }
    )

    # connect the query agent to your Weaviate collection(s)
    query_agent = QueryAgent(
        client=weaviate_client,
        collections=["Blogs"] 
    )
    return query_agent.run(query).final_answer
```

### Define model

```python
llm = OpenAI(model="gpt-4o-mini")
```

### Create Agent Workflow

```python
workflow = AgentWorkflow.from_tools_or_functions(
    [query_agent_request],
    llm=llm,
    system_prompt="You are an agent that can search a database of Weaviate blog content and answer questions about it.",
)
```

### Query Time

```python
response = await workflow.run(user_msg="How do I run Weaviate with Docker?")
print(response)
```

Python output:
```text
/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/weaviate/warnings.py:314: ResourceWarning: Con004: The connection to Weaviate was not closed properly. This can lead to memory leaks.
            Please make sure to close the connection using `client.close()`.
  warnings.warn(

To run Weaviate with Docker, follow these steps:

1. **Install Docker and Docker Compose**: Ensure that you have Docker (version 17.09.0 or higher) and Docker Compose installed. You can find installation guides for various operating systems on the Docker documentation site.

2. **Download a Weaviate Docker Image**: Use the command to pull the latest version of Weaviate:
   \```bash
   docker pull cr.weaviate.io/semitechnologies/weaviate:latest
   \```

3. **Run Weaviate**: Start a Weaviate instance using the following command:
   \```bash
   docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest
   \```
   This command will map the ports and start the Weaviate instance.

4. **Using Docker Compose**: For a more manageable configuration, it's recommended to use Docker Compose. Create a `docker-compose.yml` file with the required setup. Here’s a simple example:
   \```yaml
   version: '3.8'
   services:
     weaviate:
       image: cr.weaviate.io/semitechnologies/weaviate:latest
       ports:
         - "8080:8080"
         - "50051:50051"
       environment:
         AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
         PERSISTENCE_DATA_PATH: "/var/lib/weaviate"
   \```
   Place this file in a directory and run:
   \```bash
   docker-compose up -d
   \```

5. **Check the Status**: After starting, you can check if Weaviate is running by sending a request to its readiness endpoint:
   \```bash
   curl --fail -s localhost:8080/v1/.well-known/ready
   \```
   This command will confirm if Weaviate is up and ready for use.
```