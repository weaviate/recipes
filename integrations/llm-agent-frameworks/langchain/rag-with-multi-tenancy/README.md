# Enterprise Multi-Tenant RAG with Weaviate + LangChain

A production pattern for RAG systems serving multiple isolated tenants 
from a single Weaviate collection: the architecture behind enterprise 
internal knowledge bases.

## What this covers

- Multi tenant collection setup with per-tenant isolation
- Scoped document ingestion per tenant
- Tenant aware hybrid search (semantic + BM25)
- Minimal LangChain integration demonstrating tenant aware retrieval
- Basic handling of empty tenant queries

## When to use this pattern

Use multi tenancy when:
- Multiple teams/BUs share infrastructure but need data isolation
- You can't afford separate Weaviate instances per team
- You need tenant-level access control

## Setup

### Local (default)
1. `pip install -r requirements.txt`
2. Run the notebook top to bottom

### Optional: Production (Weaviate Cloud + OpenAI)
See the "MODE 2" section in the notebook for migration steps.

## Author

[Parikshit Sharma](https://github.com/parikshitiiitb) — 
Principal ML Engineer, production RAG systems