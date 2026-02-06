from datetime import datetime
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from typing import List, Optional
from api.dependencies import verify_api_key

router = APIRouter(prefix="/api/v1")


# -----------------------------
# Request / Response Schemas
# -----------------------------
class Message(BaseModel):
    sender: str
    text: str
    timestamp: datetime


class Metadata(BaseModel):
    channel: Optional[str] = None
    language: Optional[str] = None
    locale: Optional[str] = None


class IncomingMessage(BaseModel):
    sessionId: str
    message: Message
    conversationHistory: List[Message] = []
    metadata: Optional[Metadata] = None


class AgentResponse(BaseModel):
    sender: Optional[str] = None
    status: Optional[str] = None
    text: Optional[str] = None
    # agentActivated: bool
    # decision: str
    # confidence: float
    # agentStage: Optional[str] = None
    # signals: Optional[list] = None
    # agentReply: Optional[str] = None


# -----------------------------
# Main Ingress Endpoint
# -----------------------------


@router.post(
    "/ingest", response_model=AgentResponse, dependencies=[Depends(verify_api_key)]
)
def ingest_message(payload: IncomingMessage, request: Request):
    from agent.controller import handle_agent

    """
    Unified entry point for:
    - SMS
    - WhatsApp
    - Email
    - Chat
    Channel is intentionally NOT used for decisioning
    """

    if not payload.message.text.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    result = handle_agent(
        session_id=payload.sessionId, message=payload.message, request=request
    )

    return result
