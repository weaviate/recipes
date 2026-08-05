# Query Agent Search Mode Server

A minimal, stateless HTTP service that exposes the [Weaviate Query Agent's](https://weaviate.io/developers/agents/query) **search-only mode**. Send it a natural language query, and it returns ranked result objects from your Weaviate collections.

The service holds no state of its own: your data and indexes live in Weaviate, and query planning happens in the Query Agent service. That makes this server safe to run as a plain Kubernetes `Deployment` and scale horizontally with no coordination between replicas.

```
client ──► search-mode-server (this service, N replicas)
                 │  QueryAgent.search()
                 ▼
           Weaviate Cloud + Query Agent service
```

The parent folder contains [`search-mode-get-started.ipynb`](../search-mode-get-started.ipynb), a notebook walkthrough of the same API. Both assume a collection (default `Products`, from [`datasets/1k_products.csv`](../../../datasets/1k_products.csv)) is already loaded into your Weaviate cluster.

## API

| Endpoint | Description |
|---|---|
| `POST /search` | Run a natural language search and return ranked objects |
| `GET /healthz` | Liveness probe — process is up |
| `GET /readyz` | Readiness probe — Weaviate connection is healthy |
| `GET /docs` | Interactive OpenAPI docs (from FastAPI) |

`POST /search` request body:

```json
{
  "query": "waterproof shoes for hiking",
  "limit": 5,
  "offset": 0,
  "collections": null,
  "filtering": null,
  "diversity_weight": null
}
```

Only `query` is required. `collections` overrides the server's configured collections for one request; `filtering` biases the agent's generated filters toward `"recall"` or `"precision"`; `diversity_weight` (0.0–1.0) enables MMR reranking for more topical variety.

The response contains the ranked `objects` (uuid, collection, properties, metadata), the `searches` the agent planned from your query, and `usage.model_units` consumed.

> **Pagination note:** within a single request, a non-zero `offset` reuses the searches planned for that request, so the page is consistent. Across separate requests the agent re-plans the query, so ordering can shift slightly between pages. If you need strictly consistent pagination across requests, keep `offset` at 0 and raise `limit`, or extend the server to accept a previously returned `searches` payload.

## Configuration

All configuration is via environment variables:

| Variable | Required | Default | Description |
|---|---|---|---|
| `WEAVIATE_URL` | yes | — | Weaviate Cloud cluster URL |
| `WEAVIATE_API_KEY` | yes | — | Weaviate Cloud API key |
| `QUERY_AGENT_COLLECTIONS` | no | `Products` | Comma-separated collections the agent may search |
| `QUERY_AGENT_TIMEOUT` | no | `60` | Query Agent request timeout in seconds |

## Run locally with uv

```bash
cd query-agent/search-mode/deploy
uv sync

export WEAVIATE_URL="https://<your-cluster>.weaviate.cloud"
export WEAVIATE_API_KEY="<your-api-key>"

uv run uvicorn app.main:app --port 8080
```

Try it:

```bash
curl -s localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "waterproof shoes for hiking", "limit": 3}' | jq
```

## Build and run with Docker

The `Dockerfile` is a two-stage build: `uv sync --frozen` installs the locked dependencies into a virtualenv, which is then copied into a slim Python runtime image (no uv, no build tooling, non-root user).

```bash
docker build -t search-mode-server:0.1.0 .

docker run --rm -p 8080:8080 \
  -e WEAVIATE_URL="https://<your-cluster>.weaviate.cloud" \
  -e WEAVIATE_API_KEY="<your-api-key>" \
  search-mode-server:0.1.0
```

## Deploy to Kubernetes

The manifests in [`k8s/`](./k8s) run the server as a stateless `Deployment` (2 replicas) behind a `ClusterIP` service, with an optional `HorizontalPodAutoscaler`.

**1. Push the image** to a registry your cluster can pull from, and update `image:` in `k8s/deployment.yaml`:

```bash
docker tag search-mode-server:0.1.0 ghcr.io/<org>/search-mode-server:0.1.0
docker push ghcr.io/<org>/search-mode-server:0.1.0
```

**2. Create the credentials secret** (referenced by the Deployment via `envFrom`, so the keys become environment variables):

```bash
kubectl create secret generic weaviate-credentials \
  --from-literal=WEAVIATE_URL="https://<your-cluster>.weaviate.cloud" \
  --from-literal=WEAVIATE_API_KEY="<your-api-key>"
```

**3. Apply the manifests:**

```bash
kubectl apply -f k8s/
```

**4. Test with a port-forward:**

```bash
kubectl port-forward svc/search-mode-server 8080:80
curl -s localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "waterproof shoes for hiking", "limit": 3}' | jq
```

### How the pieces fit

- **`deployment.yaml`** — 2 replicas; credentials injected from the `weaviate-credentials` secret; `QUERY_AGENT_COLLECTIONS` set as a plain env var (change it there, or move it to a ConfigMap). The `/readyz` readiness probe checks the Weaviate connection, so a pod that loses connectivity is pulled out of the Service until it recovers. Runs as non-root with a read-only root filesystem.
- **`service.yaml`** — `ClusterIP` on port 80, for in-cluster consumers. Add an Ingress or Gateway route in front if you need external traffic.
- **`hpa.yaml`** — optional CPU-based autoscaling between 2 and 10 replicas. Because the service is stateless, scaling is just a matter of replica count; note that actual search throughput is ultimately bounded by your Weaviate cluster and Query Agent usage limits, not this tier.

## Project layout

```
search-mode/
├── search-mode-get-started.ipynb  # notebook version of the same API
└── deploy/
    ├── app/main.py                # FastAPI app: /search, /healthz, /readyz
    ├── pyproject.toml             # uv project definition
    ├── uv.lock                    # locked dependencies (used by the Docker build)
    ├── Dockerfile                 # two-stage uv build → slim runtime
    └── k8s/                       # Deployment, Service, HPA
```
