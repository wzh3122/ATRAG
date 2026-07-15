# Local Flow Evaluation

This folder contains a small, self-contained Chinese RAG retrieval evaluation.

## Data

- `corpus.jsonl`: synthetic Chinese policy-style knowledge base with distractor documents.
- `questions.csv`: factual and paraphrased questions with expected relevant document ids.

## Flows

- `bm25_baseline`: keyword/BM25 retrieval.
- `vector_baseline`: local character n-gram TF-IDF vector retrieval, or Ollama embedding if available.
- `hybrid_vector_bm25_entity`: reciprocal-rank fusion over vector, BM25, and entity/alias signals.

## Current Result

With `--local-vectorizer char_tfidf`:

| Metric | BM25 baseline | Vector baseline | Hybrid flow | Hybrid vs BM25 | Hybrid vs Vector |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recall@1 | 0.889 | 0.741 | 0.926 | +4.17% | +25.00% |
| Precision@1 | 0.889 | 0.741 | 0.926 | +4.17% | +25.00% |
| MRR | 0.944 | 0.861 | 0.963 | +1.96% | +11.83% |
| Recall@3 | 1.000 | 0.963 | 1.000 | +0.00% | +3.85% |
| MRR@3 run | 0.944 | 0.861 | 0.963 | +1.96% | +11.83% |

The top-1 gain is the most meaningful signal here: the hybrid flow places the correct document first more often, especially for paraphrased Chinese queries.

## Commands

```powershell
python atrag\evaluation\local_flow_eval\local_retrieval_ab.py --local-vectorizer char_tfidf --top-k 1 --output atrag/evaluation/local_flow_eval/report_top1.json
python atrag\evaluation\local_flow_eval\local_retrieval_ab.py --local-vectorizer char_tfidf --top-k 3 --output atrag/evaluation/local_flow_eval/report_top3.json
python atrag\evaluation\local_flow_eval\local_retrieval_ab.py --local-vectorizer char_tfidf --top-k 5 --output atrag/evaluation/local_flow_eval/report.json
```
