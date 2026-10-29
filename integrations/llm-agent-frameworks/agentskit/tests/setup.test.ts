import type { WeaviateClient } from "weaviate-client";
import { describe, expect, it, vi } from "vitest";
import { ensureChunkCollection } from "../src/setup.js";

function mockClient(exists: boolean) {
  const create = vi.fn(async () => ({}));
  const client = {
    collections: {
      create,
      exists: vi.fn(async () => exists),
    },
  } as unknown as WeaviateClient;
  return { client, create };
}

describe("ensureChunkCollection", () => {
  it("keeps an existing collection unchanged", async () => {
    const { client, create } = mockClient(true);
    await ensureChunkCollection(client, "AgentsKitTenantDocs");
    expect(create).not.toHaveBeenCalled();
  });

  it("creates a cosine collection for self-provided vectors", async () => {
    const { client, create } = mockClient(false);
    await ensureChunkCollection(client, "AgentsKitTenantDocs");
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "AgentsKitTenantDocs",
        properties: expect.arrayContaining([
          expect.objectContaining({ name: "tenant", tokenization: "field" }),
          expect.objectContaining({ name: "content" }),
        ]),
      }),
    );
  });
});
