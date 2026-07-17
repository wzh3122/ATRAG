import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from atrag.chat.history.message import StoredChatMessage
from atrag.db.models import User
from atrag.service.agent_chat_service import AgentChatService
from atrag.service.chat_completion_service import OpenAIFormatter
from atrag.views.auth import required_user

router = APIRouter(tags=["openai"])


@router.post("/chat/completions")
async def openai_chat_completions_view(request: Request, user: User = Depends(required_user)):
    """Run an agent bot and return a non-streaming OpenAI-compatible response."""
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc

    if body.get("stream"):
        raise HTTPException(status_code=422, detail="stream=true is not supported for agent bots")
    bot_id = request.query_params.get("bot_id")
    if not bot_id:
        raise HTTPException(status_code=422, detail="bot_id query parameter is required")

    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=422, detail="messages must be an array")
    user_messages = [message for message in messages if message.get("role") == "user"]
    if not user_messages or not isinstance(user_messages[-1].get("content"), str):
        raise HTTPException(status_code=422, detail="A user message with string content is required")

    result = await AgentChatService().chat_for_openai_api(
        query=user_messages[-1]["content"],
        user_id=str(user.id),
        bot_id=bot_id,
        language=body.get("language") or "en-US",
    )
    if not isinstance(result, StoredChatMessage):
        raise HTTPException(status_code=502, detail=result.get("data") or "Agent execution failed")

    references, urls = result.get_references_and_urls()
    return OpenAIFormatter.format_complete_response(
        str(uuid.uuid4()),
        result.get_main_content(),
        model=body.get("model") or "atrag",
        atrag={
            "references": references,
            "urls": urls,
            "parts": [part.model_dump(mode="json") for part in result.parts],
        },
    )
