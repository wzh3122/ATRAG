import asyncio

import pytest

from atrag.flow.base.models import SystemInput
from atrag.flow.runners.merge import MergeInput, MergeNodeRunner
from atrag.flow.runners.rerank import RerankInput, RerankNodeRunner
from atrag.query.query import DocumentWithScore


def _doc(text: str, recall_type: str, score: float = 1.0) -> DocumentWithScore:
    return DocumentWithScore(
        text=text,
        score=score,
        metadata={"recall_type": recall_type},
    )


def test_rrf_rewards_documents_recalled_by_multiple_sources():
    runner = MergeNodeRunner()
    output, _ = asyncio.run(
        runner.run(
            MergeInput(
                vector_search_docs=[
                    _doc("vector-only", "vector_search"),
                    _doc("shared", "vector_search"),
                ],
                fulltext_search_docs=[
                    _doc("shared", "fulltext_search"),
                    _doc("fulltext-only", "fulltext_search"),
                ],
            ),
            None,
        )
    )

    assert [doc.text for doc in output.docs] == ["shared", "vector-only", "fulltext-only"]
    assert output.docs[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert output.docs[0].metadata["recall_types"] == ["fulltext_search", "vector_search"]


def test_rrf_uses_rank_instead_of_raw_retrieval_score():
    runner = MergeNodeRunner()
    output, _ = asyncio.run(
        runner.run(
            MergeInput(
                vector_search_docs=[
                    _doc("vector-first", "vector_search", score=0.01),
                    _doc("vector-second", "vector_search", score=1000.0),
                ],
            ),
            None,
        )
    )

    assert [doc.text for doc in output.docs] == ["vector-first", "vector-second"]
    assert output.docs[0].score == pytest.approx(1 / 61)
    assert output.docs[1].score == pytest.approx(1 / 62)


def test_fallback_rerank_does_not_force_graph_results_first():
    docs = [
        _doc("graph", "graph_search", score=0.1),
        _doc("vector", "vector_search", score=0.3),
        _doc("fulltext", "fulltext_search", score=0.2),
    ]

    result = RerankNodeRunner()._apply_fallback_strategy(docs)

    assert [doc.text for doc in result] == ["vector", "fulltext", "graph"]


def test_graph_priority_policy_forces_graph_results_first():
    docs = [
        _doc("vector", "vector_search", score=0.3),
        _doc("graph", "graph_search", score=0.1),
        _doc("fulltext", "fulltext_search", score=0.2),
    ]

    result = RerankNodeRunner()._apply_fallback_strategy(docs, "graph_priority")

    assert [doc.text for doc in result] == ["graph", "vector", "fulltext"]


def test_standard_policy_keeps_rrf_order():
    docs = [
        _doc("graph", "graph_search", score=0.1),
        _doc("vector", "vector_search", score=0.3),
        _doc("fulltext", "fulltext_search", score=0.2),
    ]

    result = RerankNodeRunner()._apply_fallback_strategy(docs, "standard")

    assert [doc.text for doc in result] == ["vector", "fulltext", "graph"]


def test_graph_priority_recognizes_fused_graph_recall_metadata():
    fused_doc = _doc("shared", "vector_search", score=0.1)
    fused_doc.metadata["recall_types"] = ["graph_search", "vector_search"]
    docs = [
        _doc("vector", "vector_search", score=0.3),
        fused_doc,
    ]

    result = RerankNodeRunner()._apply_graph_priority(
        docs,
        "graph_priority",
    )

    assert [doc.text for doc in result] == ["shared", "vector"]


def test_graph_priority_is_applied_after_external_rerank(monkeypatch):
    docs = [
        _doc("vector", "vector_search", score=0.3),
        _doc("graph", "graph_search", score=0.1),
    ]
    runner = RerankNodeRunner()

    async def fake_actual_rerank(_ui, _si):
        return docs

    monkeypatch.setattr(runner, "_perform_actual_rerank", fake_actual_rerank)
    output, _ = asyncio.run(
        runner.run(
            RerankInput(
                use_rerank_service=True,
                model="reranker",
                model_service_provider="provider",
                custom_llm_provider="custom",
                retrieval_policy="graph_priority",
                docs=docs,
            ),
            SystemInput(
                query="张三与星海科技之间有什么关系？",
                user="user-1",
            ),
        )
    )

    assert [doc.text for doc in output.docs] == ["graph", "vector"]
