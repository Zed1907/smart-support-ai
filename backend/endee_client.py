"""
SmartSupport AI - Endee Vector Database Client
Direct HTTP implementation — no SDK dependency.

Reverse-engineered from official SDK (endee==0.1.10) source:
  create:  POST /api/v1/index/create
           body: {"index_name":str, "dim":int, "space_type":str,
                  "M":16, "ef_con":128, "checksum":-1,
                  "precision":"int8d", "version":1}

  insert:  POST /api/v1/index/{name}/vector/insert
           Content-Type: application/msgpack
           body: msgpack([[id, compressed_meta, filter_json, norm, vector], ...])
           meta: zlib(orjson(dict))
           vector is L2-normalised for cosine indexes

  search:  POST /api/v1/index/{name}/search
           body JSON: {"k":int, "ef":128, "vector":list, "include_vectors":false}
           response: msgpack([[similarity, id, meta_bytes, filter_str, norm, vector], ...])

  info:    GET  /api/v1/index/{name}/info
  list:    GET  /api/v1/index/list
  delete:  DELETE /api/v1/index/{name}/delete
"""

import logging
import zlib
import math
from typing import List, Dict, Optional

import msgpack
import orjson
import requests
from requests.adapters import HTTPAdapter, Retry

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
INDEX_NAME = "tickets"
DIMENSION  = 384
BASE_URL   = "http://127.0.0.1:8080/api/v1"

# HNSW defaults (from SDK constants.py)
DEFAULT_M       = 16
DEFAULT_EF_CON  = 128
DEFAULT_EF      = 128
PRECISION       = "int8d"
CHECKSUM        = -1

# ── HTTP session (connection-pooled, with retries) ─────────────────────────────
def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("http://",  HTTPAdapter(max_retries=retry, pool_maxsize=10))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=10))
    return s

_session: Optional[requests.Session] = None

def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = _make_session()
    return _session


# ── Compression helpers (copied from SDK compression.py) ──────────────────────
def _json_zip(data: dict) -> bytes:
    """zlib-compress a dict (SDK format for meta storage)."""
    if not data:
        return b""
    return zlib.compress(orjson.dumps(data))

def _json_unzip(data: bytes) -> dict:
    """Decompress meta bytes back to dict."""
    if not data:
        return {}
    return orjson.loads(zlib.decompress(data))


# ── Vector normalisation (SDK normalises cosine vectors before insert) ─────────
def _normalise(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-10:
        return vec
    return [x / norm for x in vec]


# ── Public API ─────────────────────────────────────────────────────────────────

def create_index(
    index_name: str = INDEX_NAME,
    dimension:  int = DIMENSION,
    space_type: str = "cosine",
) -> str:
    """
    Create a vector index.
    Idempotent — returns success even if index already exists.
    """
    url  = f"{BASE_URL}/index/create"
    body = {
        "index_name": index_name,
        "dim":        dimension,
        "space_type": space_type,
        "M":          DEFAULT_M,
        "ef_con":     DEFAULT_EF_CON,
        "checksum":   CHECKSUM,
        "precision":  PRECISION,
        "version":    1,
    }
    r = _get_session().post(url, json=body, timeout=10)

    if r.status_code == 200:
        logger.info(f"Index '{index_name}' created")
        return f"Index '{index_name}' created"
    elif r.status_code == 409:
        logger.info(f"Index '{index_name}' already exists")
        return f"Index '{index_name}' already exists"
    else:
        raise RuntimeError(f"create_index failed {r.status_code}: {r.text}")


def insert_batch(vectors: List[Dict], index_name: str = INDEX_NAME) -> str:
    """
    Insert vectors via msgpack (SDK upsert format).

    Args:
        vectors: [{"id":str, "values":[float,...], "metadata":{"team":str,"resolution":str}}]
    """
    if not vectors:
        return "Empty batch"

    batch = []
    for v in vectors:
        vec_norm = _normalise(v["values"])          # cosine → unit vector
        meta_zip = _json_zip(v.get("metadata", {})) # zlib+json compressed meta
        filter_j = orjson.dumps({}).decode()         # empty filter
        norm_val  = math.sqrt(sum(x*x for x in v["values"]))

        # SDK wire format: [id, meta_bytes, filter_str, norm, vector]
        batch.append([
            str(v["id"]),
            meta_zip,
            filter_j,
            float(norm_val),
            vec_norm,
        ])

    payload = msgpack.packb(batch, use_bin_type=True, use_single_float=True)
    url     = f"{BASE_URL}/index/{index_name}/vector/insert"
    headers = {"Content-Type": "application/msgpack"}

    r = _get_session().post(url, data=payload, headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"insert_batch failed {r.status_code}: {r.text}")

    logger.debug(f"Inserted {len(vectors)} vectors")
    return "Vectors inserted successfully"


def search(
    vector:     List[float],
    top_k:      int = 5,
    index_name: str = INDEX_NAME,
) -> dict:
    """
    Search for similar vectors.

    Returns:
        {"results": [{"id":str, "score":float, "metadata":{"team":str,"resolution":str}}, ...]}
    """
    if not vector:
        raise ValueError("Empty query vector")

    vec_norm = _normalise(vector)
    url      = f"{BASE_URL}/index/{index_name}/search"
    body     = {
        "k":               top_k,
        "ef":              DEFAULT_EF,
        "vector":          vec_norm,
        "include_vectors": False,
    }

    r = _get_session().post(url, json=body, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"search failed {r.status_code}: {r.text}")

    # Response: msgpack list of [similarity, id, meta_bytes, filter_str, norm, ...]
    raw = msgpack.unpackb(r.content, raw=False)

    results = []
    for item in raw[:top_k]:
        similarity = item[0]
        vec_id     = item[1]
        meta_bytes = item[2]
        metadata   = _json_unzip(meta_bytes) if meta_bytes else {}
        results.append({
            "id":       vec_id,
            "score":    similarity,
            "metadata": metadata,
        })

    logger.debug(f"Search: {len(results)} results")
    return {"results": results}


def list_indexes() -> list:
    """List all indexes."""
    r = _get_session().get(f"{BASE_URL}/index/list", timeout=5)
    r.raise_for_status()
    return r.json()


def check_connection() -> bool:
    """Check if Endee is reachable."""
    try:
        r = requests.get("http://localhost:8080/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False
def get_index_info(index_name: str = INDEX_NAME) -> dict:
    """
    Get metadata/info for a specific index.
    GET /api/v1/index/{name}/info
    """
    r = _get_session().get(f"{BASE_URL}/index/{index_name}/info", timeout=5)
    if r.status_code == 404:
        raise RuntimeError(f"Index '{index_name}' does not exist. Run: python ingest_tickets.py")
    r.raise_for_status()
    return r.json()


def invalidate_cache():
    """No-op: kept for API compatibility with old SDK-based version."""
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 55)
    print("  Endee Client Self-Test (direct HTTP, no SDK)")
    print("=" * 55)

    print("\n[1] Connection...")
    if not check_connection():
        print("❌  Endee not running. Start: cd ~/endee && ./build/ndd-avx2")
        exit(1)
    print("✅  Endee running")

    print("\n[2] Create index...")
    print(f"✅  {create_index()}")

    print("\n[3] Insert test vector...")
    result = insert_batch([{
        "id": "selftest_001",
        "values": [0.01] * DIMENSION,
        "metadata": {"team": "IT Support", "resolution": "Restart the service"}
    }])
    print(f"✅  {result}")

    print("\n[4] Search...")
    results = search([0.01] * DIMENSION, top_k=1)
    hits = results["results"]
    print(f"✅  {len(hits)} result(s)")
    if hits:
        h = hits[0]
        print(f"   id={h['id']}  score={h['score']:.4f}  team={h['metadata'].get('team')}")

    print("\n" + "=" * 55)