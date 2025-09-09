---
layout: recipe
toc: True
title: "Weaviate Query Agent with Crew AI"
featured: False
integration: True
agent: False
tags: ['Query Agent', 'Integration']
---
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weaviate/recipes/blob/main/integrations/llm-agent-frameworks/crewai/crewai-query-agent-as-tool.ipynb)

## Weaviate Query Agent with Crew AI

This notebook will show you how to define the Weaviate Query Agent as a tool through the Crew AI.

## Requirements
1. Weaviate Cloud instance (WCD): The Weaviate Query Agent is only accessible through WCD at the moment. You can create a serverless cluster or a free 14-day sandbox [here](https://console.weaviate.cloud/).
2. Install Crew AI with `pip install crewai`
3. Install the Weaviate Agents package with `pip install weaviate-agents`
4. You'll need a Weaviate cluster with data. If you don't have one, check out [this notebook](https://github.com/weaviate/recipes/blob/main/integrations/Weaviate-Import-Example.ipynb) to import the Weaviate Blogs.

## Import libraries and keys

```python
import weaviate
from weaviate_agents.query import QueryAgent
import os

from crewai.tools import tool
from crewai import Agent
from crewai import Task
from crewai import Crew, Process
from pydantic import BaseModel, Field
from typing import Type
from crewai.tools import BaseTool

```

Python output:
```text
/usr/local/lib/python3.11/site-packages/litellm/utils.py:149: DeprecationWarning: open_text is deprecated. Use files() instead. Refer to https://importlib-resources.readthedocs.io/en/latest/using.html#migrating-from-legacy for migration advice.
  with resources.open_text(
```
```python
os.environ["WEAVIATE_URL"] = ""
os.environ["WEAVIATE_API_KEY"] = ""
```

## Define Weaviate Query Agent as a tool

```python
class WeaviateQuerySchema(BaseModel):
    """Input for WeaviateQueryAgentTool."""

    query: str = Field(
        ...,
        description="The query to search retrieve relevant information from the Weaviate database. Pass only the query, not the question.",
    )

class WeaviateQueryAgentTool(BaseTool):
    name: str = Field(default="Weaviate Query Agent")
    description: str = Field(
        default="Send a query to the database and get the response."
    )
    args_schema: Type[BaseModel] = WeaviateQuerySchema

    def send_query_agent_request(self, query: str) -> str:
        """
        Send a query to the database and get the response.

        Args:
            query (str): The question or query to search for in the database. This can be any natural language question related to the content stored in the database.

        Returns:
            str: The response from the database containing relevant information.
        """

        weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        )
        query_agent = QueryAgent(
            client=weaviate_client,
            collections=[
                "Blogs" # we are using the Weaviate Embeddings for our Blogs collection
            ],
        )
        runner = query_agent.run(query)
        print("runner", runner)
        return runner.final_answer

    def _run(self, query: str) -> str:
        return self.send_query_agent_request(query)
    

query_agent_tool = WeaviateQueryAgentTool()
```

### 

```python
researcher = Agent(
    role="Blog Content Researcher",
    goal="Find relevant blog posts and extract key information",
    backstory="You're specialized in analyzing blog content to extract insights and answers",
    verbose=True,
    tools=[query_agent_tool]
)
```

```python
research_task = Task(
    description="Research blog posts about packaging software applications with Docker",
    expected_output="A summary of key information from relevant blog posts",
    agent=researcher
)
```

```python
blog_crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential,
    verbose=True
)
```

## Query Time

```python
result = blog_crew.kickoff()

print(result)
```

Python output:
```text
# Agent: Blog Content Researcher
## Task: Research blog posts about packaging software applications with Docker
runner original_query='packaging software applications with Docker' collection_names=['Blogs'] searches=[[QueryResultWithCollection(queries=['packaging software applications with Docker'], filters=[[]], filter_operators='AND', collection='Blogs')]] aggregations=[] usage=Usage(requests=3, request_tokens=6692, response_tokens=386, total_tokens=7078, details=None) total_time=10.477295398712158 aggregation_answer=None has_aggregation_answer=False has_search_answer=True is_partial_answer=False missing_information=[] final_answer="Docker is a platform that uses OS-level virtualization to package software applications in units called containers. These containers are similar to lightweight virtual machines, possessing their own file systems and operating system libraries, yet sharing the host system's kernel. Containers are beneficial for software application packaging as they provide considerable isolation with reduced overhead compared to traditional virtual machines. \n\nA standard Docker practice is to package a single application per container, with the container's lifecycle managed by the application's main process. If this process ends, the container typically stops. This approach ensures applications run in isolation with a consistent environment across different systems.\n\nOne of the key benefits of using Docker is its portability: as long as the Docker Engine is installed, containers can run on any OS. Docker also aids in maintaining isolation and predictability, especially for applications with complex dependencies, by encapsulating all necessary runtime dependencies within the container. This allows for easier distribution and version control via platforms like Docker Hub, facilitating seamless application upgrades and rollbacks.\n\nDocker Compose is often used in parallel with Docker to manage multi-container applications. It allows developers to define and run multi-container Docker applications in a single file, making it easier to manage complex applications that consist of multiple interacting services." sources=[Source(object_id='00a4a399-f39a-4435-b91f-7183e05ba6dd', collection='Blogs'), Source(object_id='063cb063-34cf-49ca-8c1a-c5ef9b1a89c1', collection='Blogs'), Source(object_id='cf285909-f7a3-4bd0-8810-7df41d80e20e', collection='Blogs'), Source(object_id='3757021e-f5f2-409a-9327-3b4616e78911', collection='Blogs'), Source(object_id='64183423-7f56-4b0b-8a48-6ef15cdd6bcf', collection='Blogs')]

# Agent: Blog Content Researcher
## Thought: I need to find relevant blog posts about packaging software applications with Docker. I will formulate a query to search for this topic in the Weaviate database.
## Using tool: Weaviate Query Agent
## Tool Input: 
"{\"query\": \"packaging software applications with Docker\"}"
## Tool Output: 
Docker is a platform that uses OS-level virtualization to package software applications in units called containers. These containers are similar to lightweight virtual machines, possessing their own file systems and operating system libraries, yet sharing the host system's kernel. Containers are beneficial for software application packaging as they provide considerable isolation with reduced overhead compared to traditional virtual machines. 

A standard Docker practice is to package a single application per container, with the container's lifecycle managed by the application's main process. If this process ends, the container typically stops. This approach ensures applications run in isolation with a consistent environment across different systems.

One of the key benefits of using Docker is its portability: as long as the Docker Engine is installed, containers can run on any OS. Docker also aids in maintaining isolation and predictability, especially for applications with complex dependencies, by encapsulating all necessary runtime dependencies within the container. This allows for easier distribution and version control via platforms like Docker Hub, facilitating seamless application upgrades and rollbacks.

Docker Compose is often used in parallel with Docker to manage multi-container applications. It allows developers to define and run multi-container Docker applications in a single file, making it easier to manage complex applications that consist of multiple interacting services.

/usr/local/lib/python3.11/site-packages/weaviate/warnings.py:314: ResourceWarning: Con004: The connection to Weaviate was not closed properly. This can lead to memory leaks.
            Please make sure to close the connection using `client.close()`.
  warnings.warn(

# Agent: Blog Content Researcher
## Final Answer: 
Docker is a platform that uses OS-level virtualization to package software applications in units called containers. These containers are similar to lightweight virtual machines, possessing their own file systems and operating system libraries, yet sharing the host system's kernel. Containers are beneficial for software application packaging as they provide considerable isolation with reduced overhead compared to traditional virtual machines. 

A standard Docker practice is to package a single application per container, with the container's lifecycle managed by the application's main process. If this process ends, the container typically stops. This approach ensures applications run in isolation with a consistent environment across different systems.

One of the key benefits of using Docker is its portability: as long as the Docker Engine is installed, containers can run on any OS. Docker also aids in maintaining isolation and predictability, especially for applications with complex dependencies, by encapsulating all necessary runtime dependencies within the container. This allows for easier distribution and version control via platforms like Docker Hub, facilitating seamless application upgrades and rollbacks.

Docker Compose is often used in parallel with Docker to manage multi-container applications. It allows developers to define and run multi-container Docker applications in a single file, making it easier to manage complex applications that consist of multiple interacting services.

Docker is a platform that uses OS-level virtualization to package software applications in units called containers. These containers are similar to lightweight virtual machines, possessing their own file systems and operating system libraries, yet sharing the host system's kernel. Containers are beneficial for software application packaging as they provide considerable isolation with reduced overhead compared to traditional virtual machines. 

A standard Docker practice is to package a single application per container, with the container's lifecycle managed by the application's main process. If this process ends, the container typically stops. This approach ensures applications run in isolation with a consistent environment across different systems.

One of the key benefits of using Docker is its portability: as long as the Docker Engine is installed, containers can run on any OS. Docker also aids in maintaining isolation and predictability, especially for applications with complex dependencies, by encapsulating all necessary runtime dependencies within the container. This allows for easier distribution and version control via platforms like Docker Hub, facilitating seamless application upgrades and rollbacks.

Docker Compose is often used in parallel with Docker to manage multi-container applications. It allows developers to define and run multi-container Docker applications in a single file, making it easier to manage complex applications that consist of multiple interacting services.
```