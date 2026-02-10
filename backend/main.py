"""
SmartSupport AI - Main API
Production-ready FastAPI application for ticket assignment and resolution
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from collections import Counter
import subprocess
import json
import logging
import traceback
import requests
import os

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from backend.embedder import embed_text
from backend.endee_client import search, check_connection

# ==================== LOGGING SETUP ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="SmartSupport AI",
    description="AI-powered ticket assignment & auto-resolution using Endee (RAG)",
    version="1.0.0"
)
@app.on_event("startup")
async def preload_model():
    """Load embedding model once at startup, not on first request."""
    logger.info("Preloading embedding model...")
    from backend.embedder import get_model
    get_model()
    logger.info("Embedding model ready.")

# ==================== REQUEST/RESPONSE MODELS ====================

class TicketRequest(BaseModel):
    """Request model with validation"""
    text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Ticket description (10-5000 characters)"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of similar tickets to retrieve (1-50)"
    )
    
    @validator('text')
    def text_not_empty(cls, v):
        """Ensure text is not just whitespace"""
        if not v.strip():
            raise ValueError('Ticket text cannot be empty or whitespace')
        return v.strip()


class AssignResponse(BaseModel):
    """Response model for ticket assignment"""
    predicted_team: str
    confidence: float
    similar_tickets: int
    status: str = "success"


class ResolveResponse(BaseModel):
    """Response model for ticket resolution"""
    suggested_resolution: str
    status: str = "success"


class RAGAssignResponse(BaseModel):
    """Response model for RAG-based assignment"""
    team: str
    reason: str
    status: str = "success"


class RAGResolveResponse(BaseModel):
    """Response model for RAG-based resolution"""
    resolution: str
    status: str = "success"


class HealthResponse(BaseModel):
    """Health check response"""
    api: str
    endee: bool
    ollama: bool
    model: bool


# ==================== UTILITY FUNCTIONS ====================

def call_ollama(prompt: str, timeout: int = 60) -> str:
    """
    Calls Ollama via HTTP API (faster than subprocess).
    Ollama runs as a server on port 11434.
    """
    try:
        logger.info("Calling Ollama HTTP API...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
            },
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json().get("response", "").strip()
        logger.info(f"Ollama response length: {len(result)} chars")
        return result

    except requests.exceptions.Timeout:
        logger.error("Ollama HTTP request timed out")
        return "Error: Request timed out"
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama on port 11434")
        return "Error: Ollama not reachable"
    except Exception as e:
        logger.error(f"Ollama unexpected error: {str(e)}")
        return f"Error: {str(e)}"


def majority_vote(items):
    """
    Returns most common item and its count.
    
    Args:
        items: List of items to vote on
        
    Returns:
        Tuple of (most_common_item, count)
    """
    if not items:
        return None, 0
    counter = Counter(items)
    return counter.most_common(1)[0]


def extract_metadata(item):
    """
    Safely extracts metadata from Endee search results.
    New SDK format: {"id": str, "score": float, "metadata": {"team": ..., "resolution": ...}}

    Args:
        item: Search result item

    Returns:
        Dictionary with team and resolution keys
    """
    if isinstance(item, dict):
        # New SDK format via our endee_client wrapper
        if "metadata" in item and isinstance(item["metadata"], dict):
            return item["metadata"]
        # Legacy / fallback formats
        if "payload" in item and isinstance(item["payload"], dict):
            return item["payload"]
        if "meta" in item and isinstance(item["meta"], dict):
            return item["meta"]
        return item
    return {}


def build_context(matches, max_items=5):
    """
    Builds context text for RAG from retrieved tickets.
    
    Args:
        matches: List of similar tickets from vector search
        max_items: Maximum number of tickets to include
        
    Returns:
        Formatted context string
    """
    context = ""
    for i, m in enumerate(matches[:max_items], 1):
        meta = extract_metadata(m)
        team = meta.get('team', 'Unknown')
        resolution = meta.get('resolution', 'No resolution available')
        
        # Truncate long resolutions
        if len(resolution) > 300:
            resolution = resolution[:297] + "..."
        
        context += (
            f"{i}. Team: {team}\n"
            f"   Resolution: {resolution}\n\n"
        )
    return context


def extract_json_safely(raw: str, default: dict) -> dict:
    """
    Extracts JSON from LLM response with validation.
    
    Args:
        raw: Raw LLM response text
        default: Default dict to return on failure
        
    Returns:
        Parsed JSON dict or default
    """
    try:
        # Remove markdown code blocks if present
        raw = raw.replace("```json", "").replace("```", "")
        
        # Find JSON boundaries
        start = raw.find("{")
        end = raw.rfind("}") + 1
        
        if start == -1 or end == 0:
            logger.warning("No JSON found in LLM response")
            return default
        
        # Extract and parse
        json_str = raw[start:end]
        parsed = json.loads(json_str)
        
        # Validate expected keys based on default
        if all(key in parsed for key in default.keys()):
            return parsed
        
        logger.warning(f"JSON missing expected keys: {default.keys()}")
        return default
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        return default
    except Exception as e:
        logger.error(f"Unexpected error extracting JSON: {str(e)}")
        return default


# ==================== HEALTH CHECK ====================

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Check if all system dependencies are available.
    Returns 503 if critical services are down.
    """
    health = {
        "api": "healthy",
        "endee": False,
        "ollama": False,
        "model": False
    }
    
    # Check Endee
    try:
        health["endee"] = check_connection()
    except:
        logger.warning("Endee health check failed")
    
    # Check Ollama
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=2
        )
        health["ollama"] = result.returncode == 0
    except:
        logger.warning("Ollama health check failed")
    
    # Check embedding model
    try:
        from backend.embedder import get_model
        get_model()
        health["model"] = True
    except:
        logger.warning("Embedding model health check failed")
    
    # Return 503 if critical services are down
    status = 200 if health["endee"] and health["model"] else 503
    
    return JSONResponse(content=health, status_code=status)


# ==================== CORE ENDPOINTS ====================

@app.post("/assign", response_model=AssignResponse)
def assign_ticket(req: TicketRequest):
    """
    Assign ticket to team using vector similarity and majority voting.
    
    - Embeds ticket description
    - Finds K most similar historical tickets
    - Uses majority vote to predict team
    - Returns confidence score
    """
    try:
        logger.info(f"Assignment request: {req.text[:100]}...")
        
        # Step 1: Generate embedding
        try:
            vector = embed_text(req.text)
            logger.debug(f"Generated vector of length {len(vector)}")
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Embedding generation failed: {str(e)}"
            )
        
        # Step 2: Search vector database
        try:
            result = search(vector, top_k=req.top_k)
            logger.debug(f"Search result keys: {result.keys()}")
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Endee")
            raise HTTPException(
                status_code=503,
                detail="Vector database unavailable. Please ensure Endee is running."
            )
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Vector search failed: {str(e)}"
            )
        
        # Step 3: Extract matches
        matches = result.get("results") or result.get("vectors") or []
        logger.info(f"Found {len(matches)} similar tickets")
        
        if not matches:
            logger.warning("No similar tickets found")
            return AssignResponse(
                predicted_team="Unknown",
                confidence=0.0,
                similar_tickets=0,
                status="no_matches"
            )
        
        # Step 4: Extract teams from matches
        teams = []
        for m in matches:
            meta = extract_metadata(m)
            if "team" in meta:
                teams.append(meta["team"])
        
        logger.debug(f"Extracted teams: {teams}")
        
        if not teams:
            logger.warning("No team labels found in matches")
            return AssignResponse(
                predicted_team="Unknown",
                confidence=0.0,
                similar_tickets=0,
                status="no_team_labels"
            )
        
        # Step 5: Majority voting
        team, count = majority_vote(teams)
        confidence = count / len(teams)
        
        # Apply confidence threshold
        if confidence < 0.5:
            logger.warning(f"Low confidence: {confidence:.2f}")
            team = "Manual Review"
            status = "low_confidence"
        else:
            status = "success"
        
        logger.info(f"Predicted: {team} (confidence: {confidence:.2f})")
        
        return AssignResponse(
            predicted_team=team,
            confidence=round(confidence, 2),
            similar_tickets=len(teams),
            status=status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in assign_ticket: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Internal server error during ticket assignment"
        )


@app.post("/resolve", response_model=ResolveResponse)
def resolve_ticket(req: TicketRequest):
    """
    Suggest resolution from most similar historical ticket.
    
    - Embeds ticket description
    - Finds most similar historical ticket
    - Returns its resolution
    """
    try:
        logger.info(f"Resolution request: {req.text[:100]}...")
        
        # Generate embedding
        try:
            vector = embed_text(req.text)
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Embedding generation failed: {str(e)}"
            )
        
        # Search vector database
        try:
            result = search(vector, top_k=req.top_k)
        except requests.exceptions.ConnectionError:
            raise HTTPException(
                status_code=503,
                detail="Vector database unavailable"
            )
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Vector search failed: {str(e)}"
            )
        
        # Extract matches
        matches = result.get("results") or result.get("vectors") or []
        
        if not matches:
            logger.warning("No similar tickets found")
            return ResolveResponse(
                suggested_resolution="No similar tickets found in database",
                status="no_matches"
            )
        
        # Get resolution from best match
        meta = extract_metadata(matches[0])
        resolution = meta.get("resolution", "Resolution not available")
        
        logger.info(f"Retrieved resolution: {resolution[:100]}...")
        
        return ResolveResponse(
            suggested_resolution=resolution,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in resolve_ticket: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Internal server error during resolution"
        )


@app.post("/assign-rag", response_model=RAGAssignResponse)
def assign_ticket_rag(req: TicketRequest):
    """
    RAG-based ticket assignment using Endee + Ollama.
    
    - Retrieves similar tickets from vector DB
    - Builds context from historical resolutions
    - Uses LLM to reason about team assignment
    - Returns structured JSON response
    """
    try:
        logger.info(f"RAG assignment request: {req.text[:100]}...")
        
        # Enhanced query for better retrieval
        enhanced_text = f"Support ticket: {req.text}"
        vector = embed_text(enhanced_text)
        
        # Search for similar tickets
        result = search(vector, top_k=req.top_k)
        matches = result.get("results") or result.get("vectors") or []
        
        if not matches:
            logger.warning("No similar tickets for RAG")
            return RAGAssignResponse(
                team="Unknown",
                reason="No similar historical tickets found",
                status="no_matches"
            )
        
        # Build context from matches
        context = build_context(matches)
        logger.debug(f"Built context: {len(context)} chars")
        
        # Construct prompt for LLM
        prompt = f"""You are an AI system that assigns customer support tickets to teams.

Here are similar past tickets and how they were handled:
{context}

New ticket:
"{req.text}"

Based on the similar tickets above, decide which team should handle this new ticket.

Respond ONLY with valid JSON containing these two keys:
- "team": The name of the team (e.g., "Billing and Payments", "Technical Support", etc.)
- "reason": A brief 1-2 sentence explanation

Example response format:
{{"team": "Technical Support", "reason": "Similar issues with system outages were handled by Technical Support."}}

Your JSON response:"""
        
        # Call LLM
        raw_response = call_ollama(prompt, timeout=60)
        
        if raw_response.startswith("Error:"):
            logger.error(f"LLM error: {raw_response}")
            return RAGAssignResponse(
                team="Unknown",
                reason=raw_response,
                status="llm_error"
            )
        
        # Parse JSON response
        parsed = extract_json_safely(
            raw_response,
            default={"team": "Unknown", "reason": "Failed to parse LLM response"}
        )
        
        logger.info(f"RAG predicted team: {parsed['team']}")
        
        return RAGAssignResponse(
            team=parsed["team"],
            reason=parsed["reason"],
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in assign_ticket_rag: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Internal server error during RAG assignment"
        )


@app.post("/resolve-rag", response_model=RAGResolveResponse)
def resolve_ticket_rag(req: TicketRequest):
    """
    RAG-based resolution generation using Endee + Ollama.
    
    - Retrieves similar tickets from vector DB
    - Builds context from historical resolutions
    - Uses LLM to generate tailored resolution
    - Returns natural language response
    """
    try:
        logger.info(f"RAG resolution request: {req.text[:100]}...")
        
        # Search for similar tickets
        vector = embed_text(req.text)
        result = search(vector, top_k=req.top_k)
        matches = result.get("results") or result.get("vectors") or []
        
        if not matches:
            logger.warning("No similar tickets for RAG resolution")
            return RAGResolveResponse(
                resolution="No similar past tickets found. Please contact support for assistance.",
                status="no_matches"
            )
        
        # Build context
        context = build_context(matches)
        
        # Construct prompt
        prompt = f"""You are a professional customer support assistant.

Here are past resolutions for similar tickets:
{context}

New ticket:
"{req.text}"

Based on the similar tickets above, generate a clear, helpful, and professional resolution response for this new ticket. 

Your response should:
- Address the customer's concern directly
- Be polite and empathetic
- Provide actionable steps if applicable
- Be 2-4 sentences long

Resolution:"""
        
        # Call LLM
        resolution = call_ollama(prompt, timeout=30)
        
        if resolution.startswith("Error:"):
            logger.error(f"LLM error: {resolution}")
            return RAGResolveResponse(
                resolution="Unable to generate resolution at this time. Please contact support.",
                status="llm_error"
            )
        
        logger.info(f"Generated resolution: {resolution[:100]}...")
        
        return RAGResolveResponse(
            resolution=resolution,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in resolve_ticket_rag: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Internal server error during RAG resolution"
        )


# ==================== ROOT ENDPOINT ====================

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "name": "SmartSupport AI",
        "version": "1.0.0",
        "description": "AI-powered ticket assignment & auto-resolution",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "assign": "/assign",
            "resolve": "/resolve",
            "assign_rag": "/assign-rag",
            "resolve_rag": "/resolve-rag"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)