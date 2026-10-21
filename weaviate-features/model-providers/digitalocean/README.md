# DigitalOcean Serverless Inference with Weaviate

[`digitalocean_rerank.ipynb`](./digitalocean_rerank.ipynb) runs a full retrieve → rerank → generate pipeline using only DigitalOcean Serverless Inference:

| Step | Weaviate module | Default model |
|---|---|---|
| Embeddings | `text2vec-digitalocean` | `bge-m3` |
| Reranking | `reranker-digitalocean` | `bge-reranker-v2-m3` |
| Generation (RAG) | `generative-digitalocean` | `openai-gpt-5-mini` |

## 1. Get a DigitalOcean model access key

Create one in the DigitalOcean control panel under **Serverless Inference**, and export it:

```bash
export DIGITALOCEAN_APIKEY=...
```

## 2. Build a Weaviate image with `reranker-digitalocean`

`reranker-digitalocean` is not in a released Weaviate image yet ([weaviate/weaviate#13584](https://github.com/weaviate/weaviate/pull/13584)). Build an image from that branch:

```bash
git clone https://github.com/weaviate/weaviate.git && cd weaviate
git fetch origin pull/13584/head:reranker-digitalocean && git checkout reranker-digitalocean
docker build --target weaviate -t weaviate:reranker-digitalocean .
```

Once the module is released, set `WEAVIATE_IMAGE` to the official image (e.g. `cr.weaviate.io/semitechnologies/weaviate:<version>`) and skip this step.

## 3. Start Weaviate

```bash
docker compose up -d
```

## 4. Run the notebook

```bash
pip install -U weaviate-client jupyter
jupyter notebook digitalocean_rerank.ipynb
```

The notebook passes your key to Weaviate per request in the `X-Digitalocean-Api-Key` header, so it works whether or not `DIGITALOCEAN_APIKEY` was set when the container started.

## Notes

- The rerank endpoint accepts up to 150 documents per request. Weaviate splits larger result sets into chunks automatically.
- Rerank scores are raw cross-encoder logits, so they can be negative. Only their order matters.
- List the models available to your key with:

  ```bash
  curl -s https://inference.do-ai.run/v1/models -H "Authorization: Bearer $DIGITALOCEAN_APIKEY"
  ```
