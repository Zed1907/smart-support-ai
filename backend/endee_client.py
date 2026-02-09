import requests

BASE_URL = "http://localhost:8080"
INDEX_NAME = "tickets"
DIMENSION = 384

def create_index():
    url = f"{BASE_URL}/api/v1/index/create"
    payload = {
        "index": INDEX_NAME,
        "dim": DIMENSION,
        "metric": "cosine"
    }
    r = requests.post(url, json=payload)
    try:
        return r.json()
    except Exception:
        return r.text


def insert_batch(vectors):
    url = f"{BASE_URL}/api/v1/vector/insert"
    payload = {
        "index": INDEX_NAME,
        "vectors": vectors
    }
    r = requests.post(url, json=payload)
    try:
        return r.json()
    except Exception:
        return r.text


import json

def search(vector, top_k=5):
    url = f"{BASE_URL}/api/v1/vector/search"
    payload = {
        "index": INDEX_NAME,
        "vector": vector,
        "top_k": top_k
    }

    response = requests.post(url, json=payload)

    raw = response.text.strip()

    # Try to extract JSON safely
    try:
        # Some Endee responses have extra text before JSON
        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        if first_brace != -1 and last_brace != -1:
            raw_json = raw[first_brace:last_brace + 1]
            return json.loads(raw_json)
        return {}
    except Exception:
        return {}
