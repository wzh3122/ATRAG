import pytest

from atrag.retrieval import (
    GraphIntent,
    GraphIntentDecision,
    RetrievalPolicy,
    graph_intent_classifier,
    resolve_retrieval_policy,
)


@pytest.mark.parametrize(
    "query",
    [
        "张三与星海科技之间有什么关系？",
        "A 公司如何通过子公司间接控制 B 公司？",
        "找出供应商到最终客户的多跳关系路径",
        "What is the relationship between Alice and Acme Corp?",
    ],
)
def test_classifier_detects_entity_relation_and_multi_hop_queries(query):
    decision = graph_intent_classifier.classify(query, source="test")

    assert decision.intent == GraphIntent.ENTITY_MULTI_HOP
    assert decision.confidence >= 0.80


@pytest.mark.parametrize(
    "query",
    [
        "如何提高项目交付质量？",
        "解释一下客户关系管理的基本概念",
        "星海科技成立于哪一年？",
    ],
)
def test_classifier_keeps_general_queries_on_standard_policy(query):
    decision = graph_intent_classifier.classify(query, source="test")

    assert decision.intent == GraphIntent.GENERAL


def test_agent_stage_confident_decision_overrides_upstream_decision():
    upstream = GraphIntentDecision(GraphIntent.ENTITY_MULTI_HOP, 0.95, source="upstream")
    agent = GraphIntentDecision(GraphIntent.GENERAL, 0.90, source="agent")

    assert resolve_retrieval_policy(upstream, agent) == RetrievalPolicy.STANDARD


def test_uncertain_agent_stage_falls_back_to_confident_upstream_decision():
    upstream = GraphIntentDecision(GraphIntent.ENTITY_MULTI_HOP, 0.95, source="upstream")
    agent = GraphIntentDecision(GraphIntent.UNKNOWN, 0.20, source="agent")

    assert resolve_retrieval_policy(upstream, agent) == RetrievalPolicy.GRAPH_PRIORITY


def test_decision_round_trip_preserves_structured_fields():
    decision = GraphIntentDecision(
        GraphIntent.ENTITY_MULTI_HOP,
        0.95,
        ("multi-hop signal",),
        "upstream",
    )

    assert GraphIntentDecision.from_dict(decision.to_dict()) == decision
