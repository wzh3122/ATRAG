import asyncio

from atrag.mcp import server


class _Response:
    status_code = 200
    text = ""

    def json(self):
        return {"query": "query", "items": []}


class _Client:
    def __init__(self, captured, *args, **kwargs):
        self.captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, _url, *, headers, json):
        self.captured.update(json)
        return _Response()


def _run_search(monkeypatch, query):
    captured = {}

    async def load_context(_chat_id):
        return None

    monkeypatch.setattr(server, "get_api_key", lambda: "secret")
    monkeypatch.setattr(server, "get_http_headers", lambda: {"x-atrag-chat-id": "chat-1"})
    monkeypatch.setattr(server.retrieval_policy_context_store, "load", load_context)
    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _Client(captured, *args, **kwargs),
    )

    asyncio.run(server.search_collection.fn(collection_id="collection-1", query=query))
    return captured


def test_mcp_propagates_graph_priority_policy(monkeypatch):
    payload = _run_search(monkeypatch, "张三与星海科技之间有什么关系？")

    assert payload["retrieval_policy"] == "graph_priority"


def test_mcp_propagates_standard_policy(monkeypatch):
    payload = _run_search(monkeypatch, "星海科技成立于哪一年？")

    assert payload["retrieval_policy"] == "standard"
