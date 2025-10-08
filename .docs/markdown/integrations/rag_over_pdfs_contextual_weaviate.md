---
layout: recipe
toc: True
title: "RAG over PDFs with Contextual AI Parser and Weaviate"
featured: False
integration: True
agent: False
tags: ['RAG', 'PDF Parsing', 'Document Processing', 'Contextual AI', 'Integration']
---
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weaviate/recipes/blob/main/integrations/data-platforms/contextual-ai/rag_over_pdfs_contextual_weaviate.ipynb)

# Performing RAG over PDFs with Weaviate and Contextual AI Parser
## A recipe 🧑‍🍳 🔍 💚

By Jinash Rouniyar, DevRel @ Contextual AI

**Versions used:**
- Weaviate version `1.25.3`
- Weaviate Python client `4.9.4`
- Contextual AI client `latest`
- OpenAI API (for embeddings and generation)

This is a code recipe that uses [Weaviate](https://weaviate.io/) to perform RAG over PDF documents parsed by [Contextual AI Parser](https://docs.contextual.ai/api-reference/parse/parse-file).

In this notebook, we accomplish the following:
* Parse two distinct document types using Contextual AI Parser: research papers and table-rich documents
* Extract structured markdown with document hierarchy preservation and advanced table extraction
* Generate text embeddings with OpenAI
* Perform multi-modal RAG using [Weaviate](https://weaviate.io/developers/weaviate/search/generative)

To run this notebook, you'll need:
* A [Contextual AI API key](https://docs.contextual.ai/user-guides/beginner-guide#get-your-api-key)
* An [OpenAI API key](https://platform.openai.com/docs/quickstart)

Note: This notebook can be run on any environment with internet access, as Contextual AI handles the parsing on their cloud infrastructure.

### Install Contextual AI client and Weaviate client

Note: If Colab prompts you to restart the session after running the cell below, click "restart" and proceed with running the rest of the notebook.

```python
%%capture
%pip install --upgrade --quiet contextual-client
%pip install -U weaviate-client=="4.9.4"
%pip install rich
%pip install requests

import warnings
warnings.filterwarnings("ignore")

import logging
# Suppress Weaviate client logs
logging.getLogger("weaviate").setLevel(logging.ERROR)

```

## 🔍 Part 1: Contextual AI Parser

Contextual AI Parser is a cloud-based document parsing service that excels at extracting structured information from PDFs, DOC/DOCX, and PPT/PPTX files. It provides high-quality markdown extraction with document hierarchy preservation, making it ideal for RAG applications.

The parser handles complex documents with images, tables, and hierarchical structures, providing multiple output formats including:
- `markdown-document`: Single concatenated markdown output
- `markdown-per-page`: Page-by-page markdown output
- `blocks-per-page`: Structured JSON with document hierarchy

In this notebook, we'll use Contextual AI Parser to extract structured content from two different types of documents:

1. **Research Paper**: "Attention is All You Need" - A seminal transformer architecture paper
2. **Table-Rich Document**: OmniDocBench dataset documentation with large tables

This demonstrates Contextual AI's capabilities with different document types and structures.

```python
# Documents to parse with Contextual AI
documents = [
    {
        "url": "https://arxiv.org/pdf/1706.03762",
        "title": "Attention Is All You Need",
        "type": "research_paper",
        "description": "Seminal transformer architecture paper that introduced self-attention mechanisms"
    },
    {
        "url": "https://raw.githubusercontent.com/ContextualAI/examples/refs/heads/main/03-standalone-api/04-parse/data/omnidocbench-text.pdf",
        "title": "OmniDocBench Dataset Documentation", 
        "type": "table_rich_document",
        "description": "Dataset documentation with large tables demonstrating table extraction capabilities"
    }
]

```

### API Keys Setup 🔑

We'll be using the Contextual AI API for parsing documents and OpenAI API for both generating text embeddings and for the generative model in our RAG pipeline. The code below dynamically fetches your API keys based on whether you're running this notebook in Google Colab or as a regular Jupyter notebook.

If you're running this notebook in Google Colab, make sure you [add](https://medium.com/@parthdasawant/how-to-use-secrets-in-google-colab-450c38e3ec75) your API keys as secrets.

```python
# API key variable names
contextual_api_key_var = "CONTEXTUAL_API_KEY"  # Replace with the name of your secret/env var
openai_api_key_var = "OPENAI_API_KEY"  # Replace with the name of your secret/env var

# Fetch API keys
try:
    # If running in Colab, fetch API keys from Secrets
    import google.colab
    from google.colab import userdata
    contextual_api_key = userdata.get(contextual_api_key_var)
    openai_api_key = userdata.get(openai_api_key_var)
    
    if not contextual_api_key:
        raise ValueError(f"Secret '{contextual_api_key_var}' not found in Colab secrets.")
    if not openai_api_key:
        raise ValueError(f"Secret '{openai_api_key_var}' not found in Colab secrets.")
except ImportError:
    # If not running in Colab, fetch API keys from environment variables
    import os
    contextual_api_key = os.getenv(contextual_api_key_var)
    openai_api_key = os.getenv(openai_api_key_var)
    
    if not contextual_api_key:
        raise EnvironmentError(
            f"Environment variable '{contextual_api_key_var}' is not set. "
            "Please define it before running this script."
        )
    if not openai_api_key:
        raise EnvironmentError(
            f"Environment variable '{openai_api_key_var}' is not set. "
            "Please define it before running this script."
        )

print("API keys configured successfully!")

```

### Download and parse PDFs using Contextual AI Parser

Here we use Contextual AI's Python SDK to parse a batch of PDFs. The result is structured markdown content with document hierarchy that we can use for text extraction and chunking.

```python
import requests
from contextual import ContextualAI
from time import sleep
import os

# Setup Contextual AI client
client = ContextualAI(api_key=contextual_api_key)

# Create directory for downloaded PDFs
os.makedirs("pdfs", exist_ok=True)

# Download PDFs and submit parse jobs
job_data = []

for i, doc in enumerate(documents):
    print(f"Downloading and submitting parse job for: {doc['title']}")
    print(f"Type: {doc['type']} - {doc['description']}")
    
    # Download PDF
    file_path = f"pdfs/{doc['type']}_{i}.pdf"
    with open(file_path, "wb") as f:
        f.write(requests.get(doc['url']).content)
    
    # Configure parsing parameters based on document type
    if doc['type'] == "research_paper":
        # For research papers, focus on hierarchy and figures
        parse_config = {
            "parse_mode": "standard",
            "figure_caption_mode": "concise",
            "enable_document_hierarchy": True,
            "page_range": "0-5"  # Parse first 6 pages
        }
    else:  # table_rich_document
        # For table-rich documents, enable table splitting
        parse_config = {
            "parse_mode": "standard",
            "enable_split_tables": True,
            "max_split_table_cells": 100,
        }
    
    # Submit parse job
    with open(file_path, "rb") as fp:
        response = client.parse.create(
            raw_file=fp,
            **parse_config
        )
    
    job_data.append({
        "job_id": response.job_id,
        "file_path": file_path,
        "document": doc
    })
    print(f"Submitted job {response.job_id} for {doc['title']}")

print(f"\nSubmitted {len(job_data)} parse jobs")

```

### Monitor parse job status and retrieve results

We'll monitor all parse jobs and retrieve the results once they're completed. Contextual AI provides structured markdown with document hierarchy information.

```python
# Monitor all parse jobs
completed_jobs = set()

while len(completed_jobs) &lt; len(job_data):
    for i, job_info in enumerate(job_data):
        job_id = job_info["job_id"]
        if job_id not in completed_jobs:
            status = client.parse.job_status(job_id)
            doc_title = job_info["document"]["title"]
            doc_type = job_info["document"]["type"]
            print(f"Job {i+1}/{len(job_data)} ({doc_title} - {doc_type}): {status.status}")
            
            if status.status == "completed":
                completed_jobs.add(job_id)
            elif status.status == "failed":
                print(f"Job failed for {doc_title}")
                completed_jobs.add(job_id)  # Add to completed to avoid infinite loop
    
    if len(completed_jobs) &lt; len(job_data):
        print("\nWaiting for remaining jobs to complete...")
        sleep(30)

print("\nAll parse jobs completed!")

```

### Retrieve and process parsed content

We'll retrieve the parsed results and process them into chunks suitable for vector search. Contextual AI provides excellent document structure preservation, which we'll leverage for better RAG performance.

**Key Feature**: Contextual AI preserves document hierarchy through `parent_ids`, allowing us to maintain section relationships and provide richer context to our RAG system.

```python
# Retrieve results and process into chunks
texts, titles, sources, doc_types = [], [], [], []

for job_info in job_data:
    job_id = job_info["job_id"]
    document = job_info["document"]
    
    if job_id in completed_jobs:
        try:
            print(f"Processing {document['title']} ({document['type']})")
            
            # Get results with blocks-per-page for hierarchical information
            results = client.parse.job_results(
                job_id, 
                output_types=['blocks-per-page']
            )
            
            print(f"  - {len(results.pages)} pages parsed")
            
            # Create hash table for parent content lookup
            hash_table = {}
            for page in results.pages:
                for block in page.blocks:
                    hash_table[block.id] = block.markdown
            
            # Process blocks with hierarchy context
            for page in results.pages:
                for block in page.blocks:
                    # Filter blocks based on document type and content quality
                    if (block.type in ['text', 'heading', 'table'] and 
                        len(block.markdown.strip()) > 30):
                        
                        # Add hierarchy context if available
                        context_text = block.markdown
                        
                        if hasattr(block, 'parent_ids') and block.parent_ids:
                            parent_content = "\n".join([
                                hash_table.get(parent_id, "") 
                                for parent_id in block.parent_ids
                            ])
                            if parent_content.strip():
                                context_text = f"{parent_content}\n\n{block.markdown}"
                        
                        # Add document metadata as context
                        full_text = f"Document: {document['title']}\nType: {document['type']}\n\n{context_text}"
                        
                        texts.append(full_text)
                        titles.append(document['title'])
                        sources.append(f"Page {page.index + 1}")
                        doc_types.append(document['type'])
                        
        except Exception as e:
            print(f"Error processing {document['title']}: {e}")

print(f"\nProcessed {len(texts)} chunks from {len(set(titles))} documents")
print(f"Document types: {', '.join(set(doc_types))}")

```

## 💚 Part 2: Weaviate
### Create and configure an embedded Weaviate collection

[Embedded Weaviate](https://weaviate.io/developers/weaviate/installation/embedded) allows you to spin up a Weaviate instance directly from your application code, without having to use a Docker container. If you're interested in other deployment methods, like using Docker-Compose or Kubernetes, check out this [page](https://weaviate.io/developers/weaviate/installation) in the Weaviate docs.

```python
import weaviate

# Connect to Weaviate embedded
client_weaviate = weaviate.connect_to_embedded(
    headers={
        "X-OpenAI-Api-Key": openai_api_key
    }
)

```

```python
import weaviate.classes.config as wc
from weaviate.classes.config import Property, DataType

# Define the collection name
collection_name = "contextual_parser"

# Delete the collection if it already exists
if (client_weaviate.collections.exists(collection_name)):
    client_weaviate.collections.delete(collection_name)

# Create the collection
collection = client_weaviate.collections.create(
    name=collection_name,
    vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(
        model="text-embedding-3-large",                           # Specify your embedding model here
    ),

    # Enable generative model from OpenAI
    generative_config=wc.Configure.Generative.openai(
    model="gpt-4o"                                                # Specify your generative model for RAG here
    ),

    # Define properties of metadata
    properties=[
        wc.Property(
            name="text",
            data_type=wc.DataType.TEXT
        ),
        wc.Property(
            name="title",
            data_type=wc.DataType.TEXT,
            skip_vectorization=True
        ),
        wc.Property(
            name="source",
            data_type=wc.DataType.TEXT,
            skip_vectorization=True
        ),
        wc.Property(
            name="document_type",
            data_type=wc.DataType.TEXT,
            skip_vectorization=True
        ),
    ]
)

```

### Wrangle data into an acceptable format for Weaviate

Transform our data from lists to a list of dictionaries for insertion into our Weaviate collection.

```python
# Initialize the data object
data = []

# Create a dictionary for each row by iterating through the corresponding lists
for text, title, source, doc_type in zip(texts, titles, sources, doc_types):
    data_point = {
        "text": text,
        "title": title,
        "source": source,
        "document_type": doc_type,
    }
    data.append(data_point)

print(f"Prepared {len(data)} chunks for insertion into Weaviate")
print(f"Chunks by document type:")
for doc_type in set(doc_types):
    count = doc_types.count(doc_type)
    print(f"  - {doc_type}: {count} chunks")

```

### Insert data into Weaviate and generate embeddings

Embeddings will be generated upon insertion to our Weaviate collection.

```python
# Insert text chunks and metadata into vector DB collection
response = collection.data.insert_many(
    data
)

if (response.has_errors):
    print(response.errors)
else:
    print("Insert complete.")

```

### Query the data

Here, we perform a simple similarity search to return the most similar embedded chunks to our search query.

```python
from weaviate.classes.query import MetadataQuery

# Example 1: Search for transformer-related content
print("=== Searching for Transformer Architecture ===")
response = collection.query.near_text(
    query="transformer architecture attention mechanism",
    limit=3,
    return_metadata=MetadataQuery(distance=True),
    return_properties=["text", "title", "source", "document_type"]
)

for i, obj in enumerate(response.objects):
    print(f"\n--- Result {i+1} ---")
    print(f"Title: {obj.properties['title']}")
    print(f"Type: {obj.properties['document_type']}")
    print(f"Source: {obj.properties['source']}")
    print(f"Similarity: {1 - obj.metadata.distance:.3f}")
    print(f"Text preview: {obj.properties['text'][:200]}...")

print("\n" + "="*50)

# Example 2: Search for table-related content
print("\n=== Searching for Table/Data Content ===")
response = collection.query.near_text(
    query="dataset table benchmark performance metrics",
    limit=3,
    return_metadata=MetadataQuery(distance=True),
    return_properties=["text", "title", "source", "document_type"]
)

for i, obj in enumerate(response.objects):
    print(f"\n--- Result {i+1} ---")
    print(f"Title: {obj.properties['title']}")
    print(f"Type: {obj.properties['document_type']}")
    print(f"Source: {obj.properties['source']}")
    print(f"Similarity: {1 - obj.metadata.distance:.3f}")
    print(f"Text preview: {obj.properties['text'][:200]}...")

```

### Perform RAG on parsed articles

Weaviate's `generate` module allows you to perform RAG over your embedded data without having to use a separate framework.

We specify a prompt that includes the field we want to search through in the database (in this case it's `text`), a query that includes our search term, and the number of retrieved results to use in the generation.

```python
from rich.console import Console
from rich.panel import Panel

# Example 1: RAG on Transformer Architecture
print("=== RAG Query: Transformer Architecture ===")
prompt = "Explain how {text} works, using only the retrieved context."
query = "transformer attention mechanism"

response = collection.generate.near_text(
    query=query,
    limit=4,
    grouped_task=prompt,
    return_properties=["text", "title", "source", "document_type"]
)

# Prettify the output using Rich
console = Console()
console.print(Panel(f"{prompt}".replace("{text}", query), title="Prompt", border_style="bold red"))
console.print(Panel(response.generated, title="Generated Content", border_style="bold green"))

```

```python
# Example 2: RAG on Dataset/Benchmark Information
print("\n=== RAG Query: Dataset and Benchmark Information ===")
prompt = "What information does the retrieved context provide about {text}?"
query = "dataset benchmark performance evaluation"

response = collection.generate.near_text(
    query=query,
    limit=4,
    grouped_task=prompt,
    return_properties=["text", "title", "source", "document_type"]
)

# Prettify the output using Rich
console = Console()
console.print(Panel(f"{prompt}".replace("{text}", query), title="Prompt", border_style="bold red"))
console.print(Panel(response.generated, title="Generated Content", border_style="bold green"))

```

```python
# Example 3: Cross-document comparison
print("\n=== RAG Query: Cross-Document Analysis ===")
prompt = "Compare and contrast the information about {text} from different document types in the retrieved context."
query = "attention mechanisms neural networks"

response = collection.generate.near_text(
    query=query,
    limit=4,
    grouped_task=prompt,
    return_properties=["text", "title", "source", "document_type"]
)

# Prettify the output using Rich
console = Console()
console.print(Panel(f"{prompt}".replace("{text}", query), title="Prompt", border_style="bold red"))
console.print(Panel(response.generated, title="Generated Content", border_style="bold green"))

```

## Summary

This notebook demonstrates a unique RAG pipeline using Contextual AI Parser and Weaviate with two distinct document types:

### What We Demonstrated:
1. **Research Paper Parsing**: "Attention is All You Need" with document hierarchy preservation
2. **Table-Rich Document Parsing**: OmniDocBench dataset with advanced table extraction
3. **Multi-format RAG**: Semantic search across different document types
4. **Contextual Intelligence**: Leveraging document structure for better retrieval

### Contextual AI Parser Advantages:
- **Cloud-based processing**: No local GPU/compute requirements
- **Document hierarchy preservation**: Maintains section relationships and structure
- **Advanced table handling**: Smart table splitting with header propagation
- **Multiple output formats**: Blocks, markdown, and structured JSON
- **Production-ready**: Scalable cloud service with enterprise features

### Key Differentiators from Other Parsers:
- **Hierarchical context**: Parent-child relationships preserved in chunks
- **Table intelligence**: Large tables automatically split with context preservation
- **Document type awareness**: Different parsing strategies for different content types
- **Rich metadata**: Document structure information enhances RAG quality

### Weaviate Integration Benefits:
- **Multi-modal search**: Query across different document types simultaneously
- **Metadata filtering**: Filter by document type, source, and other attributes
- **Generative AI**: Built-in RAG with context-aware generation
- **Scalability**: From embedded development to cloud production

### Next Steps for Enhancement:
* Implement document-level metadata for better source attribution
* Add hybrid search combining keyword and semantic search
* Experiment with different chunking strategies for each document type
* Use advanced RAG frameworks like [DSPy](https://weaviate.io/developers/integrations/llm-frameworks/dspy) or [LlamaIndex](https://docs.llamaindex.ai/en/stable/api_reference/storage/vector_store/weaviate/)
* Implement [Agentic RAG](https://weaviate.io/blog/what-is-agentic-rag) for complex multi-step queries

---

**Ready to get started?** This notebook provides a complete, production-ready example of integrating Contextual AI Parser with Weaviate for sophisticated RAG applications. The combination of Contextual AI's advanced parsing capabilities and Weaviate's powerful vector search and generation features creates a robust foundation for document-based AI applications.

```python
# Cleanup: Close Weaviate connection
client_weaviate.close()

# Optional: Clean up downloaded PDFs
import shutil
if os.path.exists("pdfs"):
    shutil.rmtree("pdfs")
    print("Cleaned up downloaded PDFs")

```