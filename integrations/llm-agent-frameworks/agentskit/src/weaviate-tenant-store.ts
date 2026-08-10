import type {
  RetrievedDocument,
  VectorDocument,
  VectorMemory,
  VectorSearchOptions,
} from "@agentskit/core";
import type { Collection, Properties } from "weaviate-client";
import { generateUuid5 } from "weaviate-client";

export type ChunkProperties = Properties & {
  chunkId: string;
  content: string;
  tenant: string;
  source?: string;
  documentId?: string;
  chunkIndex?: number;
  metadataJson: string;
};

export interface WeaviateTenantStoreConfig {
  collection: Collection<ChunkProperties>;
  tenant: string;
  topK?: number;
}

function nonEmpty(value: string, name: string): string {
  const normalized = value.trim();
  if (normalized.length === 0) throw new Error(`${name} must not be empty`);
  return normalized;
}

function positiveInteger(value: number | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isInteger(value) || value < 1)
    throw new Error("topK must be a positive integer");
  return value;
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === "number" ? value : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseMetadata(value: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(value);
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function similarityFromCosineDistance(
  distance: number | undefined,
): number | undefined {
  if (distance === undefined) return undefined;
  return Math.max(0, Math.min(1, 1 - distance));
}

function toRetrievedDocument(
  object: Awaited<
    ReturnType<Collection<ChunkProperties>["query"]["nearVector"]>
  >["objects"][number],
): RetrievedDocument {
  const { properties } = object;
  const tenant = optionalString(properties.tenant) ?? "";
  const documentId = optionalString(properties.documentId);
  const chunkIndex = optionalNumber(properties.chunkIndex);
  const metadata = {
    ...parseMetadata(optionalString(properties.metadataJson) ?? "{}"),
    tenant,
    documentId,
    chunkIndex,
  };

  return {
    id: optionalString(properties.chunkId) || object.uuid,
    content: optionalString(properties.content) ?? "",
    source: optionalString(properties.source),
    score: similarityFromCosineDistance(object.metadata?.distance),
    metadata,
  };
}

export function createWeaviateTenantStore(
  config: WeaviateTenantStoreConfig,
): VectorMemory {
  const tenant = nonEmpty(config.tenant, "tenant");
  const defaultTopK = positiveInteger(config.topK, 5);
  const { collection } = config;

  return {
    async store(documents: VectorDocument[]): Promise<void> {
      if (documents.length === 0) return;

      const objects = documents.map((document) => ({
        id: generateUuid5(`${tenant}:${document.id}`),
        properties: {
          chunkId: document.id,
          content: document.content,
          tenant,
          source: optionalString(document.metadata?.source),
          documentId: optionalString(document.metadata?.documentId),
          chunkIndex: optionalNumber(document.metadata?.chunkIndex),
          metadataJson: JSON.stringify(document.metadata ?? {}),
        },
        vectors: document.embedding,
      }));
      const existing = await Promise.all(
        objects.map(async (object) => ({
          object,
          exists: await collection.data.exists(object.id),
        })),
      );
      const newObjects = existing
        .filter((item) => !item.exists)
        .map((item) => item.object);
      const replacements = existing
        .filter((item) => item.exists)
        .map((item) => item.object);

      await Promise.all(
        replacements.map((object) =>
          collection.data.replace({
            id: object.id,
            properties: object.properties,
            vectors: object.vectors,
          }),
        ),
      );

      if (newObjects.length === 0) return;
      const result = await collection.data.insertMany(newObjects);

      if (result.hasErrors) {
        const messages = Object.values(result.errors).map(
          (error) => error.message,
        );
        throw new Error(
          `Weaviate rejected ${messages.length} object(s): ${messages.join("; ")}`,
        );
      }
    },

    async search(
      embedding: number[],
      options: VectorSearchOptions = {},
    ): Promise<RetrievedDocument[]> {
      if (embedding.length === 0)
        throw new Error("embedding must not be empty");

      const topK = positiveInteger(options.topK, defaultTopK);
      const threshold = options.threshold ?? 0;
      const result = await collection.query.nearVector(embedding, {
        filters: collection.filter.byProperty("tenant").equal(tenant),
        limit: topK,
        returnMetadata: ["distance"],
        returnProperties: [
          "chunkId",
          "content",
          "tenant",
          "source",
          "documentId",
          "chunkIndex",
          "metadataJson",
        ],
      });

      return result.objects
        .map(toRetrievedDocument)
        .filter(
          (document) =>
            document.score === undefined || document.score >= threshold,
        );
    },
  };
}
