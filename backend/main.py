from fastapi import FastAPI
from pydantic import BaseModel
from collections import Counter

from backend.embedder import embed_text
from backend.endee_client import search
import subprocess
import json
def call_ollama(prompt: str) -> str:
    """
    Calls Ollama locally and returns the model response as text.
    """
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()


app = FastAPI(
    title="SmartSupport AI",
    description="AI-powered ticket assignment & auto-resolution using Endee",
    version="1.0.0"
)

# --------- Request / Response Models ---------

class TicketRequest(BaseModel):
    text: str
    top_k: int = 5

class AssignResponse(BaseModel):
    predicted_team: str
    confidence: float
    similar_tickets: int

class ResolveResponse(BaseModel):
    suggested_resolution: str


# --------- Core Logic ---------

def majority_vote(items):
    counter = Counter(items)
    return counter.most_common(1)[0]
def extract_metadata(item):
    """
    Safely extracts metadata from Endee search results.
    Handles different response formats.
    """
    if isinstance(item, dict):
        if "metadata" in item and isinstance(item["metadata"], dict):
            return item["metadata"]
        if "payload" in item and isinstance(item["payload"], dict):
            return item["payload"]
        return item
    return {}
def build_context(matches, max_items=5):
    """
    Builds context text for RAG from retrieved tickets.
    """
    context = ""
    for i, m in enumerate(matches[:max_items], 1):
        meta = extract_metadata(m)
        context += (
            f"{i}. Team: {meta.get('team')}\n"
            f"   Resolution: {meta.get('resolution')}\n\n"
        )
    return context


# --------- API Endpoints ---------

@app.post("/assign", response_model=AssignResponse)
def assign_ticket(req: TicketRequest):
    vector = embed_text(req.text)
    result = search(vector, top_k=req.top_k)

    matches = (
        result.get("results")
        or result.get("vectors")
        or []
    )

    if not matches:
        return AssignResponse(
            predicted_team="Unknown",
            confidence=0.0,
            similar_tickets=0
        )

    teams = []
    for m in matches:
        meta = extract_metadata(m)
        if "team" in meta:
            teams.append(meta["team"])

    if not teams:
        return AssignResponse(
            predicted_team="Unknown",
            confidence=0.0,
            similar_tickets=0
        )

    team, count = majority_vote(teams)
    confidence = count / len(teams)

    return AssignResponse(
        predicted_team=team,
        confidence=round(confidence, 2),
        similar_tickets=len(teams)
    )


@app.post("/resolve", response_model=ResolveResponse)
def resolve_ticket(req: TicketRequest):
    vector = embed_text(req.text)
    result = search(vector, top_k=req.top_k)

    matches = (
        result.get("results")
        or result.get("vectors")
        or []
    )

    if not matches:
        return ResolveResponse(
            suggested_resolution="No similar tickets found"
        )

    # Safely extract resolution from best match
    meta = extract_metadata(matches[0])
    resolution = meta.get("resolution", "Resolution not available")

    return ResolveResponse(
        suggested_resolution=resolution
    )
@app.post("/assign-rag")
def assign_ticket_rag(req: TicketRequest):
    enhanced_text = f"Support ticket: {req.text}"
    vector = embed_text(enhanced_text)

    result = search(vector, top_k=req.top_k)
    matches = (
        result.get("results")
        or result.get("vectors")
        or []
    )

    if not matches:
        return {
            "team": "Unknown",
            "reason": "No similar historical tickets found"
        }

    context = build_context(matches)

    prompt = f"""
You are an AI system that assigns customer support tickets.

Here are similar past tickets and how they were handled:
{context}

New ticket:
"{req.text}"

Decide:
1. Which team should handle this ticket
2. One short reason

Respond ONLY in valid JSON with keys:
team, reason
"""

    raw = call_ollama(prompt)

    try:
        start = raw.find("{")
        end = raw.rfind("}")
        return json.loads(raw[start:end + 1])
    except Exception:
        return {
            "team": "Unknown",
            "reason": raw
        }
@app.post("/resolve-rag")
def resolve_ticket_rag(req: TicketRequest):
    vector = embed_text(req.text)

    result = search(vector, top_k=req.top_k)
    matches = (
        result.get("results")
        or result.get("vectors")
        or []
    )

    if not matches:
        return {
            "resolution": "No similar past tickets found"
        }

    context = build_context(matches)

    prompt = f"""
You are a professional customer support assistant.

Here are past resolutions for similar tickets:
{context}

New ticket:
"{req.text}"

Generate a clear, helpful resolution response.
"""

    resolution = call_ollama(prompt)

    return {
        "resolution": resolution
    }
