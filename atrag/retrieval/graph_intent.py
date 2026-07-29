import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Optional


class RetrievalPolicy(StrEnum):
    STANDARD = "standard"
    GRAPH_PRIORITY = "graph_priority"


class GraphIntent(StrEnum):
    ENTITY_MULTI_HOP = "entity_multi_hop"
    GENERAL = "general"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GraphIntentDecision:
    intent: GraphIntent
    confidence: float
    evidence: tuple[str, ...] = ()
    source: str = "rules"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["intent"] = self.intent.value
        data["evidence"] = list(self.evidence)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "GraphIntentDecision":
        return cls(
            intent=GraphIntent(data["intent"]),
            confidence=float(data["confidence"]),
            evidence=tuple(data.get("evidence", ())),
            source=str(data.get("source", "rules")),
        )


class GraphIntentClassifier:
    """Deterministic entity-relation classifier used outside generative prompts."""

    _STRONG_PATTERNS = (
        re.compile(
            r"多跳|多级关系|多层关系|关系链|关联链|关系路径|关联路径|最短路径|"
            r"股权穿透|间接控制|最终控制人|实际控制人|上下游关系|供应链关系|"
            r"人物关系|组织关系|调用链|依赖链|因果链|传播链"
        ),
        re.compile(
            r"\b(?:multi[- ]?hop|relationship chain|relation path|shortest path|"
            r"dependency chain|supply chain|ownership chain)\b",
            re.IGNORECASE,
        ),
    )
    _RELATION_PATTERNS = (
        re.compile(r".{1,40}(?:与|和|跟|同).{1,40}(?:之间)?(?:有什?么|的)?(?:关系|关联|联系)"),
        re.compile(r".{1,40}(?:如何|怎么|怎样)(?:通过.{0,40})?(?:影响|控制|关联|连接|依赖|传导).{1,40}"),
        re.compile(
            r"\b(?:relationship|relation|connection)\s+between\s+.{1,50}\s+and\s+.{1,50}",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bhow\s+.{1,50}\s+(?:affects?|controls?|relates?\s+to|connects?\s+to|depends?\s+on)\s+.{1,50}",
            re.IGNORECASE,
        ),
    )
    _GENERAL_PATTERNS = (
        re.compile(r"客户关系管理|人际关系|关系数据库|关系模型|函数关系|数量关系"),
        re.compile(
            r"\b(?:customer relationship management|relational database|relationship advice)\b",
            re.IGNORECASE,
        ),
    )

    def classify(self, query: str, *, source: str) -> GraphIntentDecision:
        normalized = " ".join((query or "").split())
        if not normalized:
            return GraphIntentDecision(GraphIntent.UNKNOWN, 0.0, ("empty query",), source)

        for pattern in self._GENERAL_PATTERNS:
            match = pattern.search(normalized)
            if match:
                return GraphIntentDecision(
                    GraphIntent.GENERAL,
                    0.95,
                    (f"general-domain phrase: {match.group(0)}",),
                    source,
                )

        for pattern in self._STRONG_PATTERNS:
            match = pattern.search(normalized)
            if match:
                return GraphIntentDecision(
                    GraphIntent.ENTITY_MULTI_HOP,
                    0.95,
                    (f"multi-hop signal: {match.group(0)}",),
                    source,
                )

        for pattern in self._RELATION_PATTERNS:
            match = pattern.search(normalized)
            if match:
                return GraphIntentDecision(
                    GraphIntent.ENTITY_MULTI_HOP,
                    0.88,
                    (f"entity-relation pattern: {match.group(0)}",),
                    source,
                )

        return GraphIntentDecision(
            GraphIntent.GENERAL,
            0.80,
            ("no entity-relation or multi-hop signal",),
            source,
        )


def resolve_retrieval_policy(
    upstream: Optional[GraphIntentDecision],
    agent: GraphIntentDecision,
) -> RetrievalPolicy:
    """Resolve two-stage decisions, preferring a confident agent-stage decision."""
    if agent.confidence >= 0.80:
        return (
            RetrievalPolicy.GRAPH_PRIORITY
            if agent.intent == GraphIntent.ENTITY_MULTI_HOP
            else RetrievalPolicy.STANDARD
        )

    if upstream and upstream.confidence >= 0.80:
        return (
            RetrievalPolicy.GRAPH_PRIORITY
            if upstream.intent == GraphIntent.ENTITY_MULTI_HOP
            else RetrievalPolicy.STANDARD
        )

    return RetrievalPolicy.STANDARD


graph_intent_classifier = GraphIntentClassifier()
