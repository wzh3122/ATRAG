from fastapi import APIRouter, Depends, Request

from atrag.db.models import User
from atrag.service.chat_completion_service import OpenAIFormatter
from atrag.views.auth import required_user

router = APIRouter(tags=["openai"])


@router.post("/chat/completions")
async def openai_chat_completions_view(request: Request, user: User = Depends(required_user)):
    """OpenAI-compatible chat completions endpoint - Not implemented for agent-type bots"""
    return OpenAIFormatter.format_error(
        "The /v1/chat/completions endpoint is not implemented. Please use WebSocket API for agent-type bots."
    )
