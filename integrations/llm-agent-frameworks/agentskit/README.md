# Tenant-filtered RAG with AgentsKit and Weaviate

**Versions:** Weaviate Database 1.37+, `weaviate-client` 3.14.0, `@agentskit/core` 1.12.7, `@agentskit/adapters` 0.14.2, `@agentskit/rag` 0.5.2, and `@agentskit/runtime` 0.10.14.

**Author:** [Emerson Braun](https://github.com/EmersonBraun) / [AgentsKit](https://www.agentskit.io)

An agent should never retrieve one customer's context while answering another customer's question. This TypeScript recipe makes that boundary explicit: AgentsKit handles chunking, embeddings, retrieval injection, and the agent runtime while Weaviate applies the tenant filter inside the vector query.

The model remains replaceable. The example uses `gemini-embedding-2` and OpenRouter's free router for generation, but either adapter can be exchanged without changing the Weaviate integration.

## What the recipe demonstrates

- A `VectorMemory` implementation that satisfies the stable AgentsKit contract.
- Pre-filtered vector search with `collection.filter.byProperty('tenant')`.
- Idempotent ingestion through deterministic Weaviate UUIDs.
- Cosine distance converted into the similarity score expected by AgentsKit.
- Credential-free tests that verify the tenant filter reaches Weaviate.

## Run it

Requirements: Node.js 22+, a Weaviate Cloud cluster, a Gemini API key, and an OpenRouter key.

```bash
cp .env.example .env
npm install
npm run start
```

The script creates a self-provided-vector collection when needed, ingests one policy document for `TENANT_ID`, then asks an AgentsKit runtime a question whose context is retrieved only from that tenant.

Use a different free model when needed:

```bash
OPENROUTER_MODEL=openrouter/free npm run start
```

## Verify it without credentials

```bash
npm run check
```

The tests mock only the Weaviate network boundary. They assert deterministic writes, the native tenant filter, result mapping, similarity thresholds, and invalid input handling.

## Why the adapter is small

AgentsKit's integration seam is the `VectorMemory` contract, not a database-specific abstraction. Keeping the adapter here makes Weaviate's native filtering visible and lets the same RAG and runtime code move to another vector store or model provider without rewriting the application.

## References

- [AgentsKit RAG documentation](https://www.agentskit.io/docs/packages/rag)
- [Weaviate TypeScript client](https://docs.weaviate.io/weaviate/client-libraries/typescript)
- [Weaviate filters](https://docs.weaviate.io/weaviate/search/filters)
