from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from atrag.flow.base.exceptions import ValidationError
from atrag.flow.base.models import BaseNodeRunner, SystemInput, register_node_runner
from atrag.query.query import DocumentWithScore


class MergeInput(BaseModel):
    merge_strategy: str = Field("rrf", description="How to merge results")
    deduplicate: bool = Field(True, description="Whether to deduplicate merged results")
    rrf_k: int = Field(60, ge=0, description="Rank constant used by reciprocal rank fusion")
    vector_search_docs: Optional[List[DocumentWithScore]] = Field(
        default_factory=list, description="Vector search docs"
    )
    fulltext_search_docs: Optional[List[DocumentWithScore]] = Field(
        default_factory=list, description="Fulltext search docs"
    )
    graph_search_docs: Optional[List[DocumentWithScore]] = Field(default_factory=list, description="Graph search docs")
    summary_search_docs: Optional[List[DocumentWithScore]] = Field(
        default_factory=list, description="Summary search docs"
    )
    vision_search_docs: Optional[List[DocumentWithScore]] = Field(
        default_factory=list, description="Vision search docs"
    )


class MergeOutput(BaseModel):
    docs: List[DocumentWithScore]


@register_node_runner(
    "merge",
    input_model=MergeInput,
    output_model=MergeOutput,
)
class MergeNodeRunner(BaseNodeRunner):
    @staticmethod
    def _document_key(doc: DocumentWithScore) -> str:
        return doc.text or ""

    def _rrf_fuse(
        self,
        rankings: Dict[str, List[DocumentWithScore]],
        rrf_k: int,
        deduplicate: bool,
    ) -> List[DocumentWithScore]:
        fused: Dict[str, dict] = {}

        for source, docs in rankings.items():
            seen_in_source = set()
            for rank, doc in enumerate(docs, start=1):
                document_key = self._document_key(doc)
                if deduplicate:
                    if document_key in seen_in_source:
                        continue
                    seen_in_source.add(document_key)
                    result_key = document_key
                else:
                    result_key = f"{source}:{rank}:{document_key}"

                contribution = 1.0 / (rrf_k + rank)
                entry = fused.get(result_key)
                if entry is None:
                    fused_doc = doc.model_copy(deep=True)
                    fused_doc.score = contribution
                    metadata = fused_doc.metadata or {}
                    recall_type = metadata.get("recall_type")
                    metadata["recall_types"] = [recall_type] if recall_type else []
                    fused_doc.metadata = metadata
                    fused[result_key] = {
                        "doc": fused_doc,
                        "best_rank": rank,
                        "sources": {source},
                    }
                    continue

                is_better_occurrence = rank < entry["best_rank"]
                entry["doc"].score = (entry["doc"].score or 0.0) + contribution
                entry["best_rank"] = min(entry["best_rank"], rank)
                entry["sources"].add(source)

                recall_type = (doc.metadata or {}).get("recall_type")
                recall_types = entry["doc"].metadata.setdefault("recall_types", [])
                if recall_type and recall_type not in recall_types:
                    recall_types.append(recall_type)
                    recall_types.sort()

                # Keep the payload from the strongest individual occurrence.
                if is_better_occurrence:
                    replacement = doc.model_copy(deep=True)
                    replacement.score = entry["doc"].score
                    replacement.metadata = replacement.metadata or {}
                    replacement.metadata["recall_types"] = recall_types
                    entry["doc"] = replacement

        ordered_entries = sorted(
            fused.items(),
            key=lambda item: (
                -(item[1]["doc"].score or 0.0),
                item[1]["best_rank"],
                item[0],
            ),
        )
        return [entry["doc"] for _, entry in ordered_entries]

    async def run(self, ui: MergeInput, si: SystemInput) -> Tuple[MergeOutput, dict]:
        """
        Run merge node. ui: user input; si: system input (SystemInput).
        Returns (output, system_output)
        """
        if ui.merge_strategy != "rrf":
            raise ValidationError(f"Unknown merge strategy: {ui.merge_strategy}")

        rankings = {
            "vector_search": ui.vector_search_docs or [],
            "fulltext_search": ui.fulltext_search_docs or [],
            "graph_search": ui.graph_search_docs or [],
            "summary_search": ui.summary_search_docs or [],
            "vision_search": ui.vision_search_docs or [],
        }
        docs = self._rrf_fuse(rankings, rrf_k=ui.rrf_k, deduplicate=ui.deduplicate)
        return MergeOutput(docs=docs), {}
