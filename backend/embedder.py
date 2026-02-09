from sentence_transformers import SentenceTransformer

# Lightweight, fast, 384-dim model (good for production + interviews)
_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str):
    vec = _model.encode(text, normalize_embeddings=True)
    return vec.tolist()
