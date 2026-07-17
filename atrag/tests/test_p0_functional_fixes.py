import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from atrag.chat.history.message import create_assistant_message
from atrag.db.models import ApiKey, ApiKeyStatus, Evaluation, Question
from atrag.db.repositories.api_key import AsyncApiKeyRepositoryMixin
from atrag.db.repositories.evaluation import AsyncEvaluationRepositoryMixin
from atrag.exceptions import EvaluationQuotaExceededException
from atrag.schema import view_models
from atrag.service.chat_completion_service import OpenAIFormatter
from atrag.views.internal_auth import require_internal_service
from atrag.views.openai import openai_chat_completions_view


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalars(self):
        return self

    def first(self):
        return self.value


class _SqliteBind:
    class dialect:
        name = "sqlite"


class _FakeSession:
    def __init__(self, existing=None):
        self.existing = existing
        self.added = []

    def get_bind(self):
        return _SqliteBind()

    async def execute(self, _statement):
        return _ScalarResult(self.existing)

    def add(self, value):
        self.added.append(value)

    async def flush(self):
        for value in self.added:
            if isinstance(value, ApiKey) and value.key is None:
                value.key = "sk-created"

    async def refresh(self, _value):
        return None


class _ApiKeyRepository(AsyncApiKeyRepositoryMixin):
    def __init__(self, session):
        self.session = session

    async def execute_with_transaction(self, operation):
        return await operation(self.session)


class _EvaluationRepository(AsyncEvaluationRepositoryMixin):
    def __init__(self, session):
        self.session = session

    async def execute_with_transaction(self, operation):
        return await operation(self.session)


class _Request:
    def __init__(self, body, bot_id="bot-1"):
        self.body = body
        self.query_params = {"bot_id": bot_id} if bot_id else {}

    async def json(self):
        return self.body


def test_agent_message_defaults_optional_arrays_to_empty_lists():
    message = view_models.AgentMessage(query="hello")

    assert message.collections == []
    assert message.files == []


def test_agent_message_rejects_null_files():
    with pytest.raises(ValueError):
        view_models.AgentMessage(query="hello", files=None)


def test_system_api_key_is_created_when_query_returns_no_rows():
    session = _FakeSession()
    key = asyncio.run(_ApiKeyRepository(session).get_or_create_system_api_key("user-1"))

    assert key == "sk-created"
    assert len(session.added) == 1
    assert session.added[0].is_system is True


def test_existing_system_api_key_is_reused():
    existing = ApiKey(user="user-1", key="sk-existing", status=ApiKeyStatus.ACTIVE, is_system=True)
    session = _FakeSession(existing=existing)
    key = asyncio.run(_ApiKeyRepository(session).get_or_create_system_api_key("user-1"))

    assert key == "sk-existing"
    assert session.added == []


def test_evaluation_question_limit_is_checked_before_creation():
    evaluation = Evaluation(user_id="user-1", name="eval")
    questions = [Question(question_text=f"q-{index}", ground_truth="a") for index in range(3)]

    with pytest.raises(EvaluationQuotaExceededException) as exc_info:
        asyncio.run(
            _EvaluationRepository(_FakeSession()).create_evaluation_with_limits(
                evaluation,
                questions,
                max_questions=2,
                max_active_evaluations=1,
                max_daily_items=10,
            )
        )
    assert exc_info.value.details["quota_type"] == "max_questions_per_evaluation"


def test_openai_response_keeps_agent_metadata_structured():
    response = OpenAIFormatter.format_complete_response(
        "message-id",
        "answer",
        atrag={"references": [{"document_id": "doc-1"}]},
    )

    assert response["choices"][0]["message"]["content"] == "answer"
    assert response["atrag"]["references"] == [{"document_id": "doc-1"}]
    assert "|DOC_QA_REFERENCES|" not in response["choices"][0]["message"]["content"]


def test_openai_agent_endpoint_uses_structured_result(monkeypatch):
    async def fake_chat(_self, **_kwargs):
        return create_assistant_message(
            content="answer",
            chat_id="chat-id",
            references=[{"document_id": "doc-1"}],
        )

    monkeypatch.setattr("atrag.views.openai.AgentChatService.chat_for_openai_api", fake_chat)
    response = asyncio.run(
        openai_chat_completions_view(
            _Request(
                {
                    "model": "atrag",
                    "stream": False,
                    "messages": [{"role": "user", "content": "question"}],
                }
            ),
            SimpleNamespace(id="user-1"),
        )
    )

    assert response["choices"][0]["message"]["content"] == "answer"
    assert response["atrag"]["references"] == [{"document_id": "doc-1"}]


def test_stored_agent_message_exposes_answer_and_references():
    result = create_assistant_message(
        content="answer",
        chat_id="chat-id",
        references=[{"document_id": "doc-1"}],
    )

    references, urls = result.get_references_and_urls()
    assert result.get_main_content() == "answer"
    assert references == [{"document_id": "doc-1"}]
    assert urls == []


def test_internal_service_authentication_is_fail_closed(monkeypatch):
    monkeypatch.delenv("ATRAG_INTERNAL_SERVICE_TOKEN", raising=False)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(require_internal_service("anything"))
    assert exc_info.value.status_code == 503


def test_internal_service_authentication_rejects_user_credentials(monkeypatch):
    monkeypatch.setenv("ATRAG_INTERNAL_SERVICE_TOKEN", "internal-secret")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(require_internal_service("ordinary-user-token"))
    assert exc_info.value.status_code == 401


def test_internal_service_authentication_accepts_configured_token(monkeypatch):
    monkeypatch.setenv("ATRAG_INTERNAL_SERVICE_TOKEN", "internal-secret")
    assert asyncio.run(require_internal_service("internal-secret")) is None
