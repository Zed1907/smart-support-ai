"""
SmartSupport AI - Embedding Module
Handles text-to-vector conversion using SentenceTransformers
"""

from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_model():
    """
    Lazy-load the embedding model.
    Uses LRU cache to ensure model is loaded only once.
    
    Returns:
        SentenceTransformer model instance
        
    Raises:
        ImportError: If sentence-transformers is not installed
        Exception: If model loading fails
    """
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        logger.info(f"Model loaded successfully (dim={EMBEDDING_DIM})")
        return model
    except ImportError:
        logger.error("sentence-transformers not installed")
        raise ImportError(
            "sentence-transformers package required. "
            "Install with: pip install sentence-transformers"
        )
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def embed_text(text: str, normalize: bool = True) -> list:
    """
    Convert text to embedding vector.
    
    Args:
        text: Input text to embed
        normalize: Whether to L2-normalize embeddings (recommended for cosine similarity)
        
    Returns:
        List of floats representing the embedding vector
        
    Raises:
        ValueError: If text is empty
        Exception: If embedding generation fails
        
    Example:
        >>> vector = embed_text("My payment failed")
        >>> len(vector)
        384
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")
    
    try:
        model = get_model()
        
        # Generate embedding
        vector = model.encode(
            text,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        
        # Convert numpy array to list
        return vector.tolist()
        
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise Exception(f"Failed to generate embedding: {str(e)}")


def embed_batch(texts: list, normalize: bool = True, batch_size: int = 32) -> list:
    """
    Convert multiple texts to embeddings in batches for efficiency.
    
    Args:
        texts: List of strings to embed
        normalize: Whether to L2-normalize embeddings
        batch_size: Number of texts to process at once
        
    Returns:
        List of embedding vectors (list of lists)
        
    Example:
        >>> texts = ["ticket 1", "ticket 2", "ticket 3"]
        >>> vectors = embed_batch(texts)
        >>> len(vectors)
        3
    """
    if not texts:
        raise ValueError("Cannot embed empty text list")
    
    try:
        model = get_model()
        
        # Generate embeddings in batch
        vectors = model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100
        )
        
        # Convert to list of lists
        return [v.tolist() for v in vectors]
        
    except Exception as e:
        logger.error(f"Batch embedding failed: {str(e)}")
        raise Exception(f"Failed to generate batch embeddings: {str(e)}")


def get_embedding_dimension() -> int:
    """
    Get the dimensionality of embeddings from this model.
    
    Returns:
        Integer dimension size
    """
    return EMBEDDING_DIM


if __name__ == "__main__":
    # Test the embedding module
    logging.basicConfig(level=logging.INFO)
    
    print("Testing embedding module...")
    
    # Test single embedding
    try:
        vec = embed_text("This is a test support ticket")
        print(f"✅ Single embedding: dimension={len(vec)}")
    except Exception as e:
        print(f"❌ Single embedding failed: {e}")
    
    # Test batch embedding
    try:
        texts = [
            "Payment issue",
            "Account locked",
            "Password reset needed"
        ]
        vectors = embed_batch(texts)
        print(f"✅ Batch embedding: {len(vectors)} vectors generated")
    except Exception as e:
        print(f"❌ Batch embedding failed: {e}")
    
    # Test error handling
    try:
        embed_text("")
        print("❌ Should have raised ValueError for empty text")
    except ValueError:
        print("✅ Empty text validation working")