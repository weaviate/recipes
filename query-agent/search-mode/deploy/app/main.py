"""Stateless HTTP service exposing the Weaviate Query Agent's search-only mode.

All state (data, indexes, query planning) lives in Weaviate and the Query Agent
service, so this server can be scaled horizontally without coordination.
"""

import os
from contextlib import asynccontextmanager
from typing import Literal, Optional

import weaviate
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from weaviate.agents.query import QueryAgent
from weaviate.auth import Auth


def _collections_from_env() -> list:
    raw = os.environ.get("QUERY_AGENT_COLLECTIONS", "Products")
    return [c.strip() for c in raw.split(",") if c.strip()]


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
    )
    state["client"] = client
    state["agent"] = QueryAgent(
        client=client,
        collections=_collections_from_env(),
        timeout=int(os.environ.get("QUERY_AGENT_TIMEOUT", "60")),
    )
    yield
    client.close()
    state.clear()


app = FastAPI(
    title="Query Agent Search Mode Server",
    description="Natural language search over Weaviate collections via the Query Agent's search-only mode.",
    version="0.1.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results to return")
    offset: int = Field(0, ge=0, description="Number of results to skip, for pagination")
    collections: Optional[list[str]] = Field(
        None, description="Override the collections configured via QUERY_AGENT_COLLECTIONS"
    )
    filtering: Optional[Literal["recall", "precision"]] = Field(
        None, description="Bias the agent's generated filters toward recall or precision"
    )
    diversity_weight: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="MMR reranking weight; higher values favor topical variety"
    )


_METADATA_FIELDS = ("distance", "certainty", "score", "explain_score", "rerank_score")


def _serialize_object(obj) -> dict:
    metadata = {
        field: getattr(obj.metadata, field, None)
        for field in _METADATA_FIELDS
        if getattr(obj.metadata, field, None) is not None
    }
    return {
        "uuid": str(obj.uuid),
        "collection": getattr(obj, "collection", None),
        "properties": obj.properties,
        "metadata": metadata,
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict:
    client = state.get("client")
    if client is None or not client.is_ready():
        raise HTTPException(status_code=503, detail="Weaviate connection not ready")
    return {"status": "ready"}


@app.post("/search")
def search(request: SearchRequest) -> dict:
    agent: QueryAgent = state["agent"]
    collections: Optional[list] = request.collections
    try:
        response = agent.search(
            request.query,
            limit=request.limit,
            collections=collections,
            filtering=request.filtering,
            diversity_weight=request.diversity_weight,
        )
        if request.offset:
            # next() re-executes the already-planned searches with an offset,
            # so the page comes from the same result set as the first call.
            response = response.next(limit=request.limit, offset=request.offset)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Query Agent request failed: {exc}") from exc

    return {
        "query": request.query,
        "objects": [_serialize_object(obj) for obj in response.search_results.objects],
        "searches": [s.model_dump(mode="json") for s in (response.searches or [])],
        "usage": {"model_units": response.usage.model_units},
        "total_time": response.total_time,
    }
