"""Demo queries using the Rank soft-ranking parameter on the Amazon Products dataset.

Run import_amazon_products.py first to populate the collection.

Usage:
    OPENAI_API_KEY=sk-... uv run query_with_rank.py [--depth N]
"""

import argparse
import os
from typing import Optional

import openai
import weaviate
from weaviate.classes.query import Boost, Filter, MetadataQuery


COLLECTION_NAME = "AmazonProduct"
QUERY = "calculator"


def embed_query(text: str) -> list[float]:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(input=text, model="text-embedding-3-small", dimensions=1536)
    return resp.data[0].embedding


def print_results(title: str, results) -> None:
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    for i, obj in enumerate(results.objects):
        p = obj.properties
        price = p.get("price")
        rating = p.get("average_rating")
        rating_count = p.get("rating_number")
        date = p.get("date_first_available")
        name = (p.get("title") or "")[:60]
        dist = f"dist={obj.metadata.distance:.4f}" if obj.metadata.distance is not None else ""
        score = f"score={obj.metadata.score:.4f}" if obj.metadata.score is not None else ""
        meta = " ".join(filter(None, [dist, score]))
        count_str = f" reviews={int(rating_count)}" if rating_count is not None else ""
        date_str = f" date={date.strftime('%Y-%m-%d')}" if date else ""
        print(f"  {i+1:2}. [{meta}] ${price:<8} rating={rating}{count_str}{date_str}  {name}")


def query_vector_baseline(collection, vector, limit: int) -> None:
    """Baseline: vector search with no ranking."""
    results = collection.query.near_vector(
        near_vector=vector,
        limit=limit,
        return_metadata=MetadataQuery(distance=True),
        return_properties=["title", "price", "average_rating", "main_category"],
    )
    print_results(f"Baseline: vector search '{QUERY}' (no boost)", results)


def query_boost_high_rating(collection, vector, depth: Optional[int], limit: int = 10) -> None:
    """Boost products with rating >= 4.9."""
    results = collection.query.near_vector(
        near_vector=vector,
        limit=limit,
        boost=Boost.filter(
            Filter.by_property("average_rating").greater_or_equal(4.9),
            weight=1.0,
            depth=depth,
        ),
        return_metadata=MetadataQuery(distance=True),
        return_properties=["title", "price", "average_rating", "main_category"],
    )
    print_results("Boost: prefer highly rated products (rating >= 4.9)", results)


def query_boost_price_range(collection, vector, depth: Optional[int], limit: int = 10) -> None:
    """Boost affordable products ($10-$50)."""
    results = collection.query.near_vector(
        near_vector=vector,
        limit=limit,
        boost=Boost.filter(
            Filter.by_property("price").less_than(50.0)
            & Filter.by_property("price").greater_than(10.0),
            weight=0.8,
            depth=depth,
        ),
        return_metadata=MetadataQuery(distance=True),
        return_properties=["title", "price", "average_rating", "main_category"],
    )
    print_results("Boost: prefer affordable products ($10-$50), weight=0.8", results)


def query_decay_price(collection, vector, depth: Optional[int], limit: int = 10) -> None:
    """Decay: prefer products near $25 price point."""
    results = collection.query.near_vector(
        near_vector=vector,
        limit=limit,
        boost=Boost.numeric_decay(
            "price",
            origin=20,
            scale=2,
            curve=Boost.Curve.EXPONENTIAL,
            weight=0.7,
            depth=depth,
        ),
        return_metadata=MetadataQuery(distance=True),
        return_properties=["title", "price", "average_rating", "main_category"],
    )
    print_results("Decay: prefer products priced near $20 (exp, scale=2)", results)


def query_decay_rating(collection, vector, depth: Optional[int], limit: int = 10) -> None:
    """Decay: prefer products with rating near 5.0."""
    results = collection.query.near_vector(
        near_vector=vector,
        limit=limit,
        boost=Boost.numeric_decay(
            "average_rating",
            origin=5.0,
            scale=1.0,
            curve=Boost.Curve.LINEAR,
            weight=0.6,
            depth=depth,
        ),
        return_metadata=MetadataQuery(distance=True),
        return_properties=["title", "price", "average_rating", "main_category"],
    )
    print_results("Decay: prefer top-rated products (linear, origin=5.0)", results)


def query_blend_rating_and_price(collection, vector, depth: Optional[int], limit: int = 10) -> None:
    """Blend: boost high rating AND prefer price near $30."""
    results = collection.query.near_vector(
        near_vector=vector,
        limit=limit,
        boost=Boost.blend(
            [
                Boost.filter(
                    Filter.by_property("average_rating").greater_or_equal(4.0),
                    weight=2.0,
                ),
                Boost.numeric_decay(
                    "price",
                    origin=30,
                    scale=100,
                    curve=Boost.Curve.EXPONENTIAL,
                ),
            ],
            weight=0.8,
            depth=depth,
        ),
        return_metadata=MetadataQuery(distance=True),
        return_properties=["title", "price", "average_rating", "main_category"],
    )
    print_results("Blend: highly rated (weight=2) + price near $30 (decay)", results)


def query_time_decay_newest(collection, vector, depth: Optional[int], limit: int = 10) -> None:
    """Time decay: prefer products closer to 2023-01-01."""
    results = collection.query.near_vector(
        near_vector=vector,
        limit=limit,
        boost=Boost.time_decay(
            "date_first_available",
            origin="2023-01-01T00:00:00Z",
            scale="200d",
            curve=Boost.Curve.EXPONENTIAL,
            weight=0.7,
            depth=depth,
        ),
        return_metadata=MetadataQuery(distance=True),
        return_properties=["title", "price", "average_rating", "date_first_available"],
    )
    print_results("Time decay: prefer products near 2023-01-01 (exp, scale=200d)", results)


def query_property_rating_count(collection, vector, depth: Optional[int], limit: int = 10) -> None:
    """Rank by rating_number (popularity) using log1p modifier."""
    results = collection.query.near_vector(
        near_vector=vector,
        limit=limit,
        boost=Boost.numeric_property(
            "rating_number",
            modifier=Boost.Modifier.LOG1P,
            weight=0.7,
            depth=depth,
        ),
        return_metadata=MetadataQuery(distance=True),
        return_properties=["title", "price", "average_rating", "rating_number", "main_category"],
    )
    print_results("Property: rank by rating_number (log1p)", results)


def query_blend_property_and_boost(collection, vector, depth: Optional[int], limit: int = 10) -> None:
    """Blend: boost high rating + rank by popularity."""
    results = collection.query.near_vector(
        near_vector=vector,
        limit=limit,
        boost=Boost.blend(
            [
                Boost.filter(
                    Filter.by_property("average_rating").greater_or_equal(4.0),
                    weight=2.0,
                ),
                Boost.numeric_property("rating_number", modifier=Boost.Modifier.LOG1P),
            ],
            weight=0.8,
            depth=depth,
        ),
        return_metadata=MetadataQuery(distance=True),
        return_properties=["title", "price", "average_rating", "rating_number", "main_category"],
    )
    print_results("Blend: boost rating >= 4.0 (weight=2) + popularity (log1p)", results)


# -- BM25 queries --

def query_bm25_baseline(collection, limit: int = 10) -> None:
    """BM25 keyword search with no boost."""
    results = collection.query.bm25(
        query=QUERY,
        limit=limit,
        return_metadata=MetadataQuery(score=True),
        return_properties=["title", "price", "average_rating", "main_category"],
    )
    print_results(f"BM25 '{QUERY}' (no boost)", results)


def query_bm25_with_boost(collection, depth: Optional[int], limit: int = 10) -> None:
    """BM25 keyword search + rating boost."""
    results = collection.query.bm25(
        query=QUERY,
        limit=limit,
        boost=Boost.filter(
            Filter.by_property("average_rating").greater_or_equal(4.9),
            weight=0.6,
            depth=depth,
        ),
        return_metadata=MetadataQuery(score=True),
        return_properties=["title", "price", "average_rating", "main_category"],
    )
    print_results(f"BM25 '{QUERY}' + boost rating >= 4.9", results)


# -- Hybrid queries --

def query_hybrid_baseline(collection, vector, limit: int = 10) -> None:
    """Hybrid search with no boost."""
    results = collection.query.hybrid(
        query=QUERY,
        vector=vector,
        limit=limit,
        return_metadata=MetadataQuery(score=True),
        return_properties=["title", "price", "average_rating", "main_category"],
    )
    print_results(f"Hybrid '{QUERY}' (no boost)", results)


def query_hybrid_with_boost(collection, vector, depth: Optional[int], limit: int = 10) -> None:
    """Hybrid search + price boost."""
    results = collection.query.hybrid(
        query=QUERY,
        vector=vector,
        limit=limit,
        boost=Boost.filter(
            Filter.by_property("price").less_than(20.0),
            weight=0.6,
            depth=depth,
        ),
        return_metadata=MetadataQuery(score=True),
        return_properties=["title", "price", "average_rating", "main_category"],
    )
    print_results(f"Hybrid '{QUERY}' + boost price < $20", results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo rank queries on Amazon Products")
    parser.add_argument(
        "--depth", type=int, default=None,
        help="Rescore depth for all rank queries (default: server default of 100, max 10000)",
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Number of results to return per query (default: 10)",
    )
    args = parser.parse_args()
    depth = args.depth
    limit = args.limit

    if depth is not None:
        print(f"Using rank depth={depth} for all queries")
    print(f"Returning {limit} results per query")

    print(f"Embedding query: '{QUERY}'...")
    vector = embed_query(QUERY)

    client = weaviate.connect_to_local()
    try:
        collection = client.collections.get(COLLECTION_NAME)

        # Verify data exists
        agg = collection.aggregate.over_all(total_count=True)
        print(f"Collection '{COLLECTION_NAME}' has {agg.total_count} objects")

        if agg.total_count == 0:
            print("No data found. Run import_amazon_products.py first.")
            return

        # Vector search
        query_vector_baseline(collection, vector, limit)
        query_boost_high_rating(collection, vector, depth, limit)
        query_boost_price_range(collection, vector, depth, limit)
        query_decay_price(collection, vector, depth, limit)
        query_decay_rating(collection, vector, depth, limit)
        query_blend_rating_and_price(collection, vector, depth, limit)
        query_time_decay_newest(collection, vector, depth, limit)
        query_property_rating_count(collection, vector, depth, limit)
        query_blend_property_and_boost(collection, vector, depth, limit)

        # BM25
        query_bm25_baseline(collection, limit)
        query_bm25_with_boost(collection, depth, limit)

        # Hybrid
        query_hybrid_baseline(collection, vector, limit)
        query_hybrid_with_boost(collection, vector, depth, limit)

    finally:
        client.close()


if __name__ == "__main__":
    main()
