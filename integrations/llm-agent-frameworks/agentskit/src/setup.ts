import type { WeaviateClient } from "weaviate-client";
import {
  dataType,
  tokenization,
  vectorDistances,
  vectorIndex,
  vectors,
} from "weaviate-client";
import type { ChunkProperties } from "./weaviate-tenant-store.js";

export async function ensureChunkCollection(
  client: WeaviateClient,
  collectionName: string,
): Promise<void> {
  if (await client.collections.exists(collectionName)) return;

  await client.collections.create<ChunkProperties>({
    name: collectionName,
    properties: [
      { name: "chunkId", dataType: dataType.TEXT },
      { name: "content", dataType: dataType.TEXT },
      {
        name: "tenant",
        dataType: dataType.TEXT,
        tokenization: tokenization.FIELD,
      },
      { name: "source", dataType: dataType.TEXT },
      { name: "documentId", dataType: dataType.TEXT },
      { name: "chunkIndex", dataType: dataType.INT },
      { name: "metadataJson", dataType: dataType.TEXT },
    ],
    vectorizers: vectors.selfProvided({
      vectorIndexConfig: vectorIndex.hnsw({
        distanceMetric: vectorDistances.COSINE,
      }),
    }),
  });
}
