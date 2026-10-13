import type { VectorDocument } from "@agentskit/core";
import type { Collection, DataObject } from "weaviate-client";
import { describe, expect, it, vi } from "vitest";
import {
  createWeaviateTenantStore,
  type ChunkProperties,
} from "../src/weaviate-tenant-store.js";

function batchSuccess() {
  return {
    allResponses: [],
    elapsedSeconds: 0,
    errors: {},
    hasErrors: false,
    uuids: {},
  };
}

function mockCollection() {
  const equal = vi.fn(() => ({ operator: "Equal", valueText: "acme" }));
  const byProperty = vi.fn(() => ({ equal }));
  const insertMany = vi.fn(async (_objects: DataObject<ChunkProperties>[]) =>
    batchSuccess(),
  );
  const exists = vi.fn(async () => false);
  const replace = vi.fn(
    async (_object: DataObject<ChunkProperties> & { id: string }) => undefined,
  );
  const nearVector = vi.fn(async () => ({
    objects: [
      {
        uuid: "f24af6b6-3f46-5f4b-9904-f35a0e0cf985",
        properties: {
          chunkId: "policy_chunk_0",
          content: "Refunds are available for 30 days.",
          tenant: "acme",
          source: "policy.md",
          documentId: "policy",
          chunkIndex: 0,
          metadataJson: '{"owner":"support"}',
        },
        metadata: { distance: 0.12 },
        references: undefined,
        vectors: undefined,
      },
    ],
  }));

  const collection = {
    data: { exists, insertMany, replace },
    filter: { byProperty },
    query: { nearVector },
  } as unknown as Collection<ChunkProperties>;

  return {
    byProperty,
    collection,
    equal,
    exists,
    insertMany,
    nearVector,
    replace,
  };
}

describe("createWeaviateTenantStore", () => {
  it("writes tenant-scoped objects with deterministic UUIDs", async () => {
    const { collection, exists, insertMany, replace } = mockCollection();
    const store = createWeaviateTenantStore({ collection, tenant: "acme" });
    const document: VectorDocument = {
      id: "policy_chunk_0",
      content: "Refunds are available for 30 days.",
      embedding: [0.1, 0.2],
      metadata: { source: "policy.md", documentId: "policy", chunkIndex: 0 },
    };

    await store.store([document]);
    exists.mockResolvedValueOnce(true);
    await store.store([document]);

    const first = insertMany.mock.calls[0]?.[0]?.[0];
    const second = replace.mock.calls[0]?.[0];
    expect(first?.id).toBe(second?.id);
    expect(first?.properties).toMatchObject({
      chunkId: "policy_chunk_0",
      tenant: "acme",
      source: "policy.md",
    });
    expect(first?.vectors).toEqual([0.1, 0.2]);
    expect(insertMany).toHaveBeenCalledTimes(1);
    expect(replace).toHaveBeenCalledTimes(1);
  });

  it("applies the tenant filter inside the vector query", async () => {
    const { byProperty, collection, equal, nearVector } = mockCollection();
    const store = createWeaviateTenantStore({
      collection,
      tenant: "acme",
      topK: 4,
    });

    const documents = await store.search([0.1, 0.2]);

    expect(byProperty).toHaveBeenCalledWith("tenant");
    expect(equal).toHaveBeenCalledWith("acme");
    expect(nearVector).toHaveBeenCalledWith(
      [0.1, 0.2],
      expect.objectContaining({
        filters: { operator: "Equal", valueText: "acme" },
        limit: 4,
        returnMetadata: ["distance"],
      }),
    );
    expect(documents).toEqual([
      expect.objectContaining({
        id: "policy_chunk_0",
        content: "Refunds are available for 30 days.",
        score: 0.88,
        metadata: expect.objectContaining({ tenant: "acme", owner: "support" }),
      }),
    ]);
  });

  it("enforces similarity thresholds after converting cosine distance", async () => {
    const { collection } = mockCollection();
    const store = createWeaviateTenantStore({ collection, tenant: "acme" });

    await expect(store.search([0.1, 0.2], { threshold: 0.9 })).resolves.toEqual(
      [],
    );
  });

  it("skips empty batches and reports Weaviate batch errors", async () => {
    const { collection, insertMany } = mockCollection();
    const store = createWeaviateTenantStore({ collection, tenant: "acme" });
    await store.store([]);
    expect(insertMany).not.toHaveBeenCalled();

    insertMany.mockResolvedValueOnce({
      ...batchSuccess(),
      hasErrors: true,
      errors: {
        0: {
          message: "invalid vector",
          object: { collection: "AgentsKitTenantDocs" },
        },
      },
    });

    await expect(
      store.store([{ id: "bad", content: "bad", embedding: [0.1] }]),
    ).rejects.toThrow("Weaviate rejected 1 object(s): invalid vector");
  });

  it("rejects unsafe empty tenant and embedding inputs", async () => {
    const { collection } = mockCollection();
    expect(() =>
      createWeaviateTenantStore({ collection, tenant: "  " }),
    ).toThrow("tenant must not be empty");

    const store = createWeaviateTenantStore({ collection, tenant: "acme" });
    await expect(store.search([])).rejects.toThrow(
      "embedding must not be empty",
    );
    await expect(store.search([0.1], { topK: 0 })).rejects.toThrow(
      "topK must be a positive integer",
    );
  });
});
