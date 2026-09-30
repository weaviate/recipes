# deepface + Weaviate

[deepface](https://github.com/serengil/deepface) is a lightweight face recognition and facial attribute analysis library for Python. It can register face embeddings in a database, then search and identify faces against them. Weaviate is one of its database backends.

`deepface_with_weaviate.ipynb` runs the integration end to end against a local Weaviate. Each step checks its result with `assert`, so running the notebook is also a test:

| Step | What it shows |
|---|---|
| Register | face embeddings stored in a Weaviate collection, duplicates skipped |
| Search | 1:N search, exact scan and HNSW (`ann`) giving the same matches |
| Identify | 1:1 verification against a registered face |
| Inside Weaviate | the collection deepface creates and what each object holds |
| Connection options | self-hosted, Weaviate Cloud, auth, timeouts, your own client |
| Multi-tenancy | one isolated gallery of faces per tenant |
| Quantization | RQ-compressed vectors with unchanged distances |
| What reached Weaviate | no GraphQL requests, and the `X-Weaviate-Client-Integration` header |

> **Note:** the Weaviate v4 integration (gRPC, no GraphQL, tenants and quantization) is not in a deepface release yet. The notebook's setup cell and [requirements.txt](./requirements.txt) both show how to install the modified library.

**Versions used:** Weaviate 1.39.7 · weaviate-client 4.23.1 · deepface 0.0.102 with the v4 integration · Facenet + OpenCV detector · Python 3.11

**Author:** Duda Nogueira

## Run it

```sh
docker compose up -d

python -m venv .venv && source .venv/bin/activate
pip install -e /path/to/deepface        # the modified library, see requirements.txt
pip install -r requirements.txt jupyter

jupyter notebook deepface_with_weaviate.ipynb
```

The first run downloads the Facenet weights (about 90 MB) to `~/.deepface`, and a few face images from deepface's test dataset to `./images`.

If a port is taken, set `WEAVIATE_HTTP_PORT`, `WEAVIATE_GRPC_PORT`, `WEAVIATE_DEBUG_PORT` or `WEAVIATE_METRICS_PORT` for both `docker compose` and the notebook. Tear everything down with `docker compose down -v`.

To run it without opening Jupyter:

```sh
jupyter nbconvert --to notebook --execute deepface_with_weaviate.ipynb --output executed.ipynb
```
