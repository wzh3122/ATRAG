import json
import logging
from typing import Optional

from atrag.db.redis_manager import RedisConnectionManager
from atrag.retrieval.graph_intent import GraphIntentDecision

logger = logging.getLogger(__name__)


class RetrievalPolicyContextStore:
    _KEY_PREFIX = "atrag:retrieval-policy:"

    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds

    async def save(self, chat_id: str, decision: GraphIntentDecision) -> None:
        if not chat_id:
            return
        try:
            client = await RedisConnectionManager.get_async_client()
            await client.set(
                f"{self._KEY_PREFIX}{chat_id}",
                json.dumps(decision.to_dict(), ensure_ascii=False),
                ex=self.ttl_seconds,
            )
        except Exception as exc:
            logger.warning(
                "Could not persist upstream graph intent; agent-stage classification will be used: %s",
                type(exc).__name__,
            )

    async def load(self, chat_id: Optional[str]) -> Optional[GraphIntentDecision]:
        if not chat_id:
            return None
        try:
            client = await RedisConnectionManager.get_async_client()
            raw = await client.get(f"{self._KEY_PREFIX}{chat_id}")
            if not raw:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return GraphIntentDecision.from_dict(json.loads(raw))
        except Exception as exc:
            logger.warning(
                "Could not load upstream graph intent; agent-stage classification will be used: %s",
                type(exc).__name__,
            )
            return None


retrieval_policy_context_store = RetrievalPolicyContextStore()
