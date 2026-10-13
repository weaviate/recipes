import { geminiEmbedder, openrouter } from "@agentskit/adapters";
import { createRAG } from "@agentskit/rag";
import { createRuntime } from "@agentskit/runtime";
import weaviate from "weaviate-client";
import { ensureChunkCollection } from "./setup.js";
import {
  createWeaviateTenantStore,
  type ChunkProperties,
} from "./weaviate-tenant-store.js";

function requiredEnvironment(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing required environment variable: ${name}`);
  return value;
}

const url = requiredEnvironment("WEAVIATE_URL");
const weaviateApiKey = requiredEnvironment("WEAVIATE_API_KEY");
const geminiApiKey = requiredEnvironment("GEMINI_API_KEY");
const openrouterApiKey = requiredEnvironment("OPENROUTER_API_KEY");
const collectionName =
  process.env.WEAVIATE_COLLECTION?.trim() || "AgentsKitTenantDocs";
const tenant = process.env.TENANT_ID?.trim() || "acme";

const client = await weaviate.connectToWeaviateCloud(url, {
  authCredentials: new weaviate.ApiKey(weaviateApiKey),
});

try {
  await ensureChunkCollection(client, collectionName);
  const collection = client.collections.use<string, ChunkProperties>(
    collectionName,
  );
  const rag = createRAG({
    embed: geminiEmbedder({
      apiKey: geminiApiKey,
      model: "gemini-embedding-2",
    }),
    store: createWeaviateTenantStore({ collection, tenant, topK: 4 }),
    chunkSize: 240,
    chunkOverlap: 24,
  });

  await rag.ingest([
    {
      id: "support-policy",
      source: "support-policy.md",
      content:
        "Acme customers can request a refund within 30 days of purchase.",
      metadata: { owner: "support" },
    },
  ]);

  const runtime = createRuntime({
    adapter: openrouter({
      apiKey: openrouterApiKey,
      model: process.env.OPENROUTER_MODEL?.trim() || "openrouter/free",
    }),
    retriever: rag,
    systemPrompt: `Answer only from retrieved documents for tenant ${tenant}. Say when context is missing.`,
  });

  const answer = await runtime.run("What is the refund window?");
  console.log(answer.content);
} finally {
  await client.close();
}
