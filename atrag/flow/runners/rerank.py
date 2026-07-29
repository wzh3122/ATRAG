import logging
import re
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from atrag.db.ops import async_db_ops
from atrag.flow.base.models import BaseNodeRunner, SystemInput, register_node_runner
from atrag.llm.llm_error_types import (
    InvalidConfigurationError,
    ProviderNotFoundError,
    RerankError,
)
from atrag.llm.rerank.rerank_service import RerankService
from atrag.query.query import DocumentWithScore

logger = logging.getLogger(__name__)


class RerankInput(BaseModel):
    use_rerank_service: bool = Field(default=True, description="Whether to use rerank service or fallback strategy")
    model: Optional[str] = Field(default=None, description="Rerank model name")
    model_service_provider: Optional[str] = Field(default=None, description="Model service provider")
    custom_llm_provider: Optional[str] = Field(
        default=None, description="Custom LLM provider (e.g., 'jina_ai', 'openai')"
    )
    docs: List[DocumentWithScore]


class RerankOutput(BaseModel):
    docs: List[DocumentWithScore]


@register_node_runner(
    "rerank",
    input_model=RerankInput,
    output_model=RerankOutput,
)
class RerankNodeRunner(BaseNodeRunner):
    _GRAPH_PRIORITY_PATTERNS = (
        re.compile(
            r"多跳|多级关系|多层关系|关系链|关联链|关系路径|关联路径|最短路径|"
            r"股权穿透|间接控制|最终控制人|实际控制人|上下游关系|供应链关系|"
            r"人物关系|组织关系|调用链|依赖链|因果链|传播链"
        ),
        re.compile(r".{1,40}(?:与|和|跟|同).{1,40}(?:之间)?(?:有什?么|的)?(?:关系|关联|联系)"),
        re.compile(r".{1,40}(?:如何|怎么|怎样)(?:通过.{0,40})?(?:影响|控制|关联|连接|依赖|传导).{1,40}"),
        re.compile(
            r"\b(?:multi[- ]?hop|relationship chain|relation path|shortest path|"
            r"dependency chain|supply chain|ownership chain)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:relationship|relation|connection)\s+between\s+.{1,50}\s+and\s+.{1,50}",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bhow\s+.{1,50}\s+(?:affects?|controls?|relates?\s+to|connects?\s+to|depends?\s+on)\s+.{1,50}",
            re.IGNORECASE,
        ),
    )

    async def run(self, ui: RerankInput, si: SystemInput) -> Tuple[RerankOutput, dict]:
        """
        Smart rerank node:
        - use_rerank_service=False: directly use fallback strategy
        - use_rerank_service=True: try rerank service, fallback on failure
        """
        docs = ui.docs

        if not docs:
            logger.info("No documents to rerank, returning empty result")
            return RerankOutput(docs=[]), {}

        # Strategy 1: If not using rerank service, directly use fallback strategy
        if not ui.use_rerank_service:
            logger.info("Rerank service disabled, using fallback strategy")
            result = self._apply_fallback_strategy(docs, si.query)
            return RerankOutput(docs=result), {}

        # Strategy 2: Try to use rerank service
        try:
            # Check configuration completeness
            if not self._is_rerank_config_valid(ui):
                logger.info("Rerank service configuration incomplete, using fallback strategy")
                result = self._apply_fallback_strategy(docs, si.query)
                return RerankOutput(docs=result), {}

            # Execute actual rerank
            result = await self._perform_actual_rerank(ui, si)
            result = self._apply_graph_priority(result, si.query)
            logger.info(f"Successfully reranked {len(result)} documents using rerank service")
            return RerankOutput(docs=result), {}

        except (InvalidConfigurationError, ProviderNotFoundError) as e:
            logger.warning(f"Rerank service configuration error, using fallback strategy: {str(e)}")
            result = self._apply_fallback_strategy(docs, si.query)
            return RerankOutput(docs=result), {}

        except RerankError as e:
            logger.warning(f"Rerank service operation failed, using fallback strategy: {str(e)}")
            result = self._apply_fallback_strategy(docs, si.query)
            return RerankOutput(docs=result), {}

        except Exception as e:
            logger.error(f"Unexpected error during rerank service, using fallback strategy: {str(e)}")
            result = self._apply_fallback_strategy(docs, si.query)
            return RerankOutput(docs=result), {}

    def _is_rerank_config_valid(self, ui: RerankInput) -> bool:
        """Check if rerank service configuration is valid"""
        return (
            ui.model
            and ui.model.strip()
            and ui.model_service_provider
            and ui.model_service_provider.strip()
            and ui.custom_llm_provider
            and ui.custom_llm_provider.strip()
        )

    async def _perform_actual_rerank(self, ui: RerankInput, si: SystemInput) -> List[DocumentWithScore]:
        """Execute actual rerank operation"""
        query = si.query
        docs = ui.docs

        # Validate configuration
        if not ui.model_service_provider:
            raise InvalidConfigurationError(
                "model_service_provider", ui.model_service_provider, "Model service provider cannot be empty"
            )

        if not ui.model:
            raise InvalidConfigurationError("model", ui.model, "Model name cannot be empty")

        if not ui.custom_llm_provider:
            raise InvalidConfigurationError(
                "custom_llm_provider", ui.custom_llm_provider, "Custom LLM provider cannot be empty"
            )

        # Get API key and base_url
        api_key = await async_db_ops.query_provider_api_key(ui.model_service_provider, si.user)
        if not api_key:
            raise InvalidConfigurationError(
                "api_key", api_key, f"API KEY not found for LLM Provider:{ui.model_service_provider}"
            )

        try:
            llm_provider = await async_db_ops.query_llm_provider_by_name(ui.model_service_provider)
            if not llm_provider:
                raise ProviderNotFoundError(ui.model_service_provider, "Rerank")
            base_url = llm_provider.base_url
        except Exception as e:
            logger.error(f"Failed to query LLM provider '{ui.model_service_provider}': {str(e)}")
            raise ProviderNotFoundError(ui.model_service_provider, "Rerank") from e

        if not base_url:
            raise InvalidConfigurationError(
                "base_url", base_url, f"Base URL not configured for provider '{ui.model_service_provider}'"
            )

        # Create and execute rerank service
        rerank_service = RerankService(
            rerank_provider=ui.custom_llm_provider,
            rerank_model=ui.model,
            rerank_service_url=base_url,
            rerank_service_api_key=api_key,
        )

        rerank_service.validate_configuration()

        logger.info(
            f"Using rerank service with provider: {ui.model_service_provider}, "
            f"model: {ui.model}, url: {base_url}, max_docs: {rerank_service.max_documents}"
        )

        return await rerank_service.async_rerank(query, docs)

    @classmethod
    def _requires_graph_priority(cls, query: str) -> bool:
        normalized_query = " ".join((query or "").split())
        return bool(normalized_query) and any(
            pattern.search(normalized_query) for pattern in cls._GRAPH_PRIORITY_PATTERNS
        )

    @staticmethod
    def _is_graph_result(doc: DocumentWithScore) -> bool:
        metadata = doc.metadata or {}
        recall_types = metadata.get("recall_types") or []
        return metadata.get("recall_type") == "graph_search" or "graph_search" in recall_types

    def _apply_graph_priority(
        self,
        docs: List[DocumentWithScore],
        query: str,
    ) -> List[DocumentWithScore]:
        if not self._requires_graph_priority(query):
            return docs

        graph_results = [doc for doc in docs if self._is_graph_result(doc)]
        if not graph_results:
            return docs

        other_results = [doc for doc in docs if not self._is_graph_result(doc)]
        logger.info(
            "Prioritized %d graph results for entity relationship or multi-hop query",
            len(graph_results),
        )
        return graph_results + other_results

    def _apply_fallback_strategy(
        self,
        docs: List[DocumentWithScore],
        query: str = "",
    ) -> List[DocumentWithScore]:
        """
        Apply fallback rerank strategy:
        Sort all results by their fused score in descending order.
        """
        if not docs:
            return docs

        result = sorted(docs, key=lambda x: x.score if x.score is not None else 0.0, reverse=True)
        result = self._apply_graph_priority(result, query)
        logger.info("Applied fallback rerank strategy: %d results sorted by fused score", len(result))
        return result
