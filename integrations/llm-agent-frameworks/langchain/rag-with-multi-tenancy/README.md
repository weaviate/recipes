# Enterprise Multi-Tenant RAG with Weaviate + LangChain

A production pattern for RAG systems serving multiple isolated tenants 
from a single Weaviate collection — the architecture behind enterprise 
internal knowledge bases.

## What this covers

- Multi-tenant collection setup with per-tenant isolation
- Scoped document ingestion per tenant
- Tenant-aware hybrid search (semantic + BM25)
- LangChain retriever with tenant context
- Graceful handling of empty tenants

## When to use this pattern

Use multi-tenancy when:
- Multiple teams/BUs share infrastructure but need data isolation
- You can't afford separate Weaviate instances per team
- You need tenant-level access control

## Setup

1. Create a free Weaviate Cloud cluster at console.weaviate.cloud
2. Copy `.env.example` to `.env` and fill in your credentials
3. `pip install -r requirements.txt`
4. Run the notebook top to bottom

## Author

[Parikshit Sharma](https://github.com/parikshitiiitb) — 
Principal ML Engineer, production RAG systems
