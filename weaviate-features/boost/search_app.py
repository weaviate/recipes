"""
Boost Search Demo — FastAPI app showcasing Weaviate's Boost parameter.

Run with:
    OPENAI_API_KEY=sk-... uv run --extra app search_app.py

Requires:
    - Local Weaviate running (docker compose up -d)
    - AmazonProduct collection populated via import_amazon_products.py
    - OPENAI_API_KEY environment variable set
"""

import json
import os
import time
from typing import Optional

import openai
import uvicorn
import weaviate
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from weaviate.classes.query import Boost, Filter, MetadataQuery

COLLECTION_NAME = "AmazonProduct"

PROFILES = {
    "vector": {
        "name": "Vector Search",
        "description": "Pure semantic similarity — no boosting applied.",
        "icon": "🔍",
    },
    "boost_rated": {
        "name": "Boost Highly Rated",
        "description": "Promote products with rating ≥ 4.9 to the top.",
        "icon": "⭐",
    },
    "boost_affordable": {
        "name": "Boost Affordable",
        "description": "Prefer products in the $10–$50 range.",
        "icon": "💰",
    },
    "decay_price": {
        "name": "Price Near $20",
        "description": "Exponential decay from $20 — closer = higher score.",
        "icon": "🎯",
    },
    "decay_cheap": {
        "name": "Cheapest First",
        "description": "Linear decay from $0 — cheaper products score higher.",
        "icon": "📉",
    },
    "popularity": {
        "name": "Most Popular",
        "description": "Rank by number of ratings (log1p scaled).",
        "icon": "🔥",
    },
    "boost_date": {
        "name": "Boost Date",
        "description": "Time decay — products closer to 2023-01-01 rank higher.",
        "icon": "🕐",
    },
    "blend": {
        "name": "Blend: Quality + Value",
        "description": "Boost rating ≥ 4.0 (weight 2) + decay price near $30.",
        "icon": "⚡",
    },
    "blend_popular_cheap": {
        "name": "Blend: Popular + Cheap",
        "description": "Popularity (log1p) + price decay from $15.",
        "icon": "🏆",
    },
}

app = FastAPI()
client: Optional[weaviate.WeaviateClient] = None
oai: Optional[openai.OpenAI] = None


def get_client() -> weaviate.WeaviateClient:
    global client
    if client is None:
        client = weaviate.connect_to_local()
    return client


def get_openai() -> openai.OpenAI:
    global oai
    if oai is None:
        oai = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return oai


def embed_query(text: str) -> list[float]:
    resp = get_openai().embeddings.create(input=text, model="text-embedding-3-small", dimensions=1536)
    return resp.data[0].embedding


def build_boost(profile: str, weight: float, depth: int = 100, origin: Optional[str] = None):
    if profile == "vector":
        return None
    if profile == "boost_rated":
        return Boost.filter(
            Filter.by_property("average_rating").greater_or_equal(4.9),
            weight=weight, depth=depth,
        )
    if profile == "boost_affordable":
        return Boost.filter(
            Filter.by_property("price").greater_than(10.0)
            & Filter.by_property("price").less_than(50.0),
            weight=weight, depth=depth,
        )
    if profile == "decay_price":
        return Boost.numeric_decay(
            "price", origin=20, scale=2, curve=Boost.Curve.EXPONENTIAL,
            weight=weight, depth=depth,
        )
    if profile == "decay_cheap":
        return Boost.numeric_decay(
            "price", origin=0, scale=30, curve=Boost.Curve.LINEAR,
            weight=weight, depth=depth,
        )
    if profile == "popularity":
        return Boost.numeric_property(
            "rating_number", modifier=Boost.Modifier.LOG1P,
            weight=weight, depth=depth,
        )
    if profile == "boost_date":
        return Boost.time_decay(
            "date_first_available",
            origin=origin or "2023-01-01T00:00:00Z",
            scale="200d",
            curve=Boost.Curve.EXPONENTIAL,
            weight=weight, depth=depth,
        )
    if profile == "blend":
        return Boost.blend(
            [
                Boost.filter(
                    Filter.by_property("average_rating").greater_or_equal(4.0), weight=2.0
                ),
                Boost.numeric_decay("price", origin=30, scale=100, curve=Boost.Curve.EXPONENTIAL),
            ],
            weight=weight, depth=depth,
        )
    if profile == "blend_popular_cheap":
        return Boost.blend(
            [
                Boost.numeric_property("rating_number", modifier=Boost.Modifier.LOG1P, weight=1.5),
                Boost.numeric_decay("price", origin=15, scale=40, curve=Boost.Curve.LINEAR),
            ],
            weight=weight, depth=depth,
        )
    return None


def do_search(query: str, profile: str, weight: float = 0.5, depth: int = 100, origin: Optional[str] = None, limit: int = 20):
    collection = get_client().collections.get(COLLECTION_NAME)
    vector = embed_query(query)
    boost = build_boost(profile, weight, depth, origin)

    t0 = time.perf_counter()
    results = collection.query.near_vector(
        near_vector=vector,
        limit=limit,
        boost=boost,
        return_metadata=MetadataQuery(distance=True),
        return_properties=[
            "title", "price", "average_rating", "rating_number", "main_category", "image", "date_first_available",
        ],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    items = []
    for obj in results.objects:
        p = obj.properties
        items.append({
            "title": p.get("title", ""),
            "price": p.get("price"),
            "rating": p.get("average_rating"),
            "reviews": int(p.get("rating_number") or 0),
            "category": p.get("main_category", ""),
            "image": p.get("image", ""),
            "date": p.get("date_first_available"),
            "distance": round(obj.metadata.distance, 4) if obj.metadata.distance else None,
        })
    return items, elapsed_ms


@app.get("/api/search")
def api_search(
    q: str = Query(...),
    profile: str = Query("vector"),
    weight: float = Query(0.5),
    depth: int = Query(100),
    origin: Optional[str] = Query(None),
    limit: int = Query(20),
):
    items, elapsed_ms = do_search(q, profile, weight, depth, origin, limit)
    return {"items": items, "elapsed_ms": round(elapsed_ms, 1), "profile": profile, "weight": weight}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    profiles_json = json.dumps(PROFILES)
    return HTMLResponse(PAGE_HTML.replace("__PROFILES_JSON__", profiles_json))


PAGE_HTML = """\
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Boost Search Demo</title>
<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config = {
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        surface: '#0f1117',
        card: '#1a1d27',
        accent: '#6366f1',
        'accent-light': '#818cf8',
      }
    }
  }
}
</script>
<style>
  body { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
  .profile-btn.active { border-color: #6366f1; background: rgba(99,102,241,0.12); }
  .profile-btn:hover:not(.active) { background: rgba(255,255,255,0.04); }
  .result-row:hover { background: rgba(99,102,241,0.06); }
  .spinner { border: 2px solid rgba(255,255,255,0.1); border-top-color: #6366f1;
             border-radius: 50%; width: 20px; height: 20px; animation: spin 0.6s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  input::placeholder { color: #555; }
</style>
</head>
<body class="bg-surface text-gray-200 min-h-screen">

<div class="flex min-h-screen">
  <!-- Sidebar -->
  <aside class="w-72 border-r border-gray-800 p-5 flex flex-col gap-2 shrink-0 overflow-y-auto">
    <h2 class="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-3">Boost Profiles</h2>
    <div id="profiles" class="flex flex-col gap-1.5"></div>
    <div class="mt-auto pt-6 text-xs text-gray-600 leading-relaxed">
      <p class="mb-2">Each profile applies a different <code class="text-accent-light">Boost</code> configuration to the same vector search.</p>
      <p>Data: Amazon Products 2023</p>
    </div>
  </aside>

  <!-- Main -->
  <main class="flex-1 flex flex-col">
    <!-- Search bar -->
    <header class="border-b border-gray-800 p-5">
      <div class="max-w-5xl mx-auto flex gap-3 items-center">
        <div class="relative flex-1">
          <input id="search" type="text" placeholder="Search products…"
            class="w-full bg-card border border-gray-700 rounded-lg px-4 py-2.5 text-sm
                   focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent
                   transition-colors" />
        </div>
        <button id="searchBtn" onclick="doSearch()"
          class="bg-accent hover:bg-accent-light text-white px-5 py-2.5 rounded-lg text-sm font-medium
                 transition-colors flex items-center gap-2">
          Search
        </button>
        <div id="spinner" class="spinner hidden"></div>
      </div>
      <!-- Weight slider -->
      <div id="weightRow" class="max-w-5xl mx-auto mt-3 flex items-center gap-3 hidden">
        <label class="text-xs text-gray-500 shrink-0 w-20">Boost weight</label>
        <input id="weightSlider" type="range" min="0" max="1" step="0.01" value="0.50"
          class="flex-1 h-1.5 accent-accent bg-gray-700 rounded-full appearance-none cursor-pointer
                 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5
                 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:bg-accent
                 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer" />
        <span id="weightValue" class="text-sm font-mono text-accent-light w-10 text-right">0.50</span>
      </div>
      <!-- Origin date slider (Boost Newest only) -->
      <div id="originRow" class="max-w-5xl mx-auto mt-3 flex items-center gap-3 hidden">
        <label class="text-xs text-gray-500 shrink-0 w-20">Boost after</label>
        <input id="originSlider" type="range" min="0" max="77" step="1" value="77"
          class="flex-1 h-1.5 accent-accent bg-gray-700 rounded-full appearance-none cursor-pointer
                 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5
                 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:bg-accent
                 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer" />
        <span id="originValue" class="text-sm font-mono text-accent-light w-24 text-right">Now</span>
      </div>
      <!-- Depth slider -->
      <div id="depthRow" class="max-w-5xl mx-auto mt-3 flex items-center gap-3 hidden">
        <label class="text-xs text-gray-500 shrink-0 w-20">Depth</label>
        <input id="depthSlider" type="range" min="10" max="10000" step="10" value="100"
          class="flex-1 h-1.5 accent-accent bg-gray-700 rounded-full appearance-none cursor-pointer
                 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5
                 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:bg-accent
                 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer" />
        <span id="depthValue" class="text-sm font-mono text-accent-light w-16 text-right">100</span>
      </div>
    </header>

    <!-- Results -->
    <div class="flex-1 overflow-y-auto p-5">
      <div class="max-w-5xl mx-auto">
        <div id="meta" class="text-xs text-gray-500 mb-3"></div>
        <div id="results"></div>
        <div id="empty" class="text-gray-600 text-center py-20">
          Type a query and select a boost profile to see results.
        </div>
      </div>
    </div>
  </main>
</div>

<script>
const PROFILES = __PROFILES_JSON__;
let activeProfile = 'vector';
let lastVector = null;  // cache embedding to avoid re-embedding on weight change
let sliderDebounce = null;

function getWeight() {
  return parseFloat(document.getElementById('weightSlider').value);
}

function updateWeightDisplay() {
  document.getElementById('weightValue').textContent = getWeight().toFixed(2);
}

function showWeightSlider(show) {
  document.getElementById('weightRow').classList.toggle('hidden', !show);
}

// Origin slider: 0 = Jan 2020, 77 = Jun 2026 ("now")
function sliderToDate(val) {
  const v = parseInt(val);
  if (v >= 77) return null; // "now"
  const year = 2020 + Math.floor(v / 12);
  const month = v % 12;  // 0-indexed
  return new Date(year, month, 1);
}

function originLabel(val) {
  const d = sliderToDate(val);
  if (!d) return 'Now';
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short' });
}

function getOriginISO() {
  const val = document.getElementById('originSlider').value;
  const d = sliderToDate(val);
  return d ? d.toISOString() : null;
}

function updateOriginDisplay() {
  document.getElementById('originValue').textContent = originLabel(document.getElementById('originSlider').value);
}

function showOriginSlider(show) {
  document.getElementById('originRow').classList.toggle('hidden', !show);
}

function getDepth() {
  return parseInt(document.getElementById('depthSlider').value);
}

function updateDepthDisplay() {
  document.getElementById('depthValue').textContent = getDepth().toLocaleString();
}

function showDepthSlider(show) {
  document.getElementById('depthRow').classList.toggle('hidden', !show);
}

function initProfiles() {
  const container = document.getElementById('profiles');
  for (const [key, p] of Object.entries(PROFILES)) {
    const btn = document.createElement('button');
    btn.className = 'profile-btn flex items-start gap-3 p-3 rounded-lg border border-transparent text-left transition-all';
    btn.dataset.key = key;
    btn.innerHTML = `
      <span class="text-lg leading-none mt-0.5">${p.icon}</span>
      <div>
        <div class="text-sm font-medium text-gray-200">${p.name}</div>
        <div class="text-xs text-gray-500 mt-0.5">${p.description}</div>
      </div>`;
    btn.onclick = () => selectProfile(key);
    container.appendChild(btn);
  }
  selectProfile('vector');
}

function selectProfile(key) {
  activeProfile = key;
  showWeightSlider(key !== 'vector');
  showOriginSlider(key === 'boost_date');
  document.querySelectorAll('.profile-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.key === key);
  });
  const q = document.getElementById('search').value.trim();
  if (q) doSearch();
}

function updateURL(q) {
  const url = new URL(window.location);
  if (q) { url.searchParams.set('q', q); } else { url.searchParams.delete('q'); }
  history.replaceState(null, '', url);
}

async function doSearch() {
  const q = document.getElementById('search').value.trim();
  if (!q) return;
  updateURL(q);

  document.getElementById('spinner').classList.remove('hidden');
  document.getElementById('searchBtn').disabled = true;
  document.getElementById('empty').classList.add('hidden');
  document.getElementById('results').innerHTML = '';
  document.getElementById('meta').textContent = '';

  try {
    const w = getWeight();
    const o = getOriginISO();
    const d = getDepth();
    let url = `/api/search?q=${encodeURIComponent(q)}&profile=${activeProfile}&weight=${w}&depth=${d}&limit=20`;
    if (o) url += `&origin=${encodeURIComponent(o)}`;
    const resp = await fetch(url);
    const data = await resp.json();
    renderResults(data);
  } catch (e) {
    document.getElementById('results').innerHTML = `<p class="text-red-400 text-sm">${e.message}</p>`;
  } finally {
    document.getElementById('spinner').classList.add('hidden');
    document.getElementById('searchBtn').disabled = false;
  }
}

function renderResults(data) {
  const profileInfo = PROFILES[data.profile];
  const weightStr = data.profile !== 'vector' ? ` · weight=${data.weight.toFixed(2)}` : '';
  document.getElementById('meta').textContent =
    `${data.items.length} results in ${data.elapsed_ms}ms — ${profileInfo.name}${weightStr}`;

  const container = document.getElementById('results');
  if (data.items.length === 0) {
    container.innerHTML = '<p class="text-gray-500 text-center py-10">No results found.</p>';
    return;
  }

  container.innerHTML = data.items.map((item, i) => `
    <div class="result-row flex items-center gap-4 py-4 px-3 rounded-lg transition-colors ${i > 0 ? 'border-t border-gray-800/50' : ''}">
      <span class="text-xs text-gray-600 w-5 text-right shrink-0">${i + 1}</span>
      <img src="${esc(item.image)}" alt="" loading="lazy"
        class="w-28 h-28 rounded-lg object-contain bg-white p-2 shrink-0"
        onerror="this.style.display='none'" />
      <div class="flex-1 min-w-0 basis-2/3">
        <div class="text-base font-medium text-gray-100">${esc(item.title)}</div>
        <div class="text-xs text-gray-500 mt-1">${esc(item.category)}${item.date ? ' · ' + new Date(item.date).toLocaleDateString('en-US', {year:'numeric',month:'short',day:'numeric'}) : ''}</div>
      </div>
      <div class="text-right shrink-0 flex flex-col items-end gap-0.5">
        <span class="text-sm font-semibold text-green-400">$${item.price != null ? item.price.toFixed(2) : '—'}</span>
        <span class="text-xs text-yellow-500">${'★'.repeat(Math.round(item.rating || 0))}${'☆'.repeat(5 - Math.round(item.rating || 0))}
          <span class="text-gray-500 ml-1">${item.rating?.toFixed(1) ?? '—'}</span></span>
        <span class="text-xs text-gray-600">${item.reviews.toLocaleString()} reviews</span>
      </div>
      <span class="text-xs text-gray-700 w-16 text-right shrink-0">${item.distance ?? ''}</span>
    </div>
  `).join('');
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

document.getElementById('search').addEventListener('keydown', e => {
  if (e.key === 'Enter') doSearch();
});

document.getElementById('weightSlider').addEventListener('input', () => {
  updateWeightDisplay();
  clearTimeout(sliderDebounce);
  sliderDebounce = setTimeout(() => {
    const q = document.getElementById('search').value.trim();
    if (q) doSearch();
  }, 250);
});

document.getElementById('originSlider').addEventListener('input', () => {
  updateOriginDisplay();
  clearTimeout(sliderDebounce);
  sliderDebounce = setTimeout(() => {
    const q = document.getElementById('search').value.trim();
    if (q) doSearch();
  }, 250);
});

document.getElementById('depthSlider').addEventListener('input', () => {
  updateDepthDisplay();
  clearTimeout(sliderDebounce);
  sliderDebounce = setTimeout(() => {
    const q = document.getElementById('search').value.trim();
    if (q) doSearch();
  }, 250);
});

initProfiles();

// Restore query from URL and auto-search
const params = new URLSearchParams(window.location.search);
const initialQ = params.get('q');
if (initialQ) {
  document.getElementById('search').value = initialQ;
  doSearch();
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
