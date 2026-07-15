#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_corpus(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["relevant_doc_ids"] = [x.strip() for x in row["relevant_doc_ids"].split("|") if x.strip()]
    return rows


def bm25_scores(query: str, docs: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
    tokenized_docs = [tokenize(doc["title"] + " " + doc["text"]) for doc in docs]
    doc_freq = defaultdict(int)
    for toks in tokenized_docs:
        for tok in set(toks):
            doc_freq[tok] += 1
    avgdl = sum(len(toks) for toks in tokenized_docs) / max(len(tokenized_docs), 1)
    query_terms = tokenize(query)
    scores = {}
    for doc, toks in zip(docs, tokenized_docs):
        counts = Counter(toks)
        score = 0.0
        dl = len(toks)
        for term in query_terms:
            if counts[term] == 0:
                continue
            idf = math.log(1 + (len(docs) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = counts[term] + k1 * (1 - b + b * dl / avgdl)
            score += idf * counts[term] * (k1 + 1) / denom
        scores[doc["id"]] = score
    return scores


def entity_scores(query: str, docs: list[dict[str, Any]]) -> dict[str, float]:
    scores = {}
    for doc in docs:
        score = 0.0
        for entity in doc.get("entities", []) + doc.get("aliases", []):
            if entity.lower() in query.lower():
                score += 2.0
            else:
                q_tokens = set(tokenize(query))
                e_tokens = set(tokenize(entity))
                if q_tokens & e_tokens:
                    score += 0.4
        scores[doc["id"]] = score
    return scores


def ollama_embed(texts: list[str], base_url: str, model: str) -> list[list[float]]:
    vectors = []
    session = requests.Session()
    for text in texts:
        payload = {"model": model, "input": text}
        resp = session.post(f"{base_url.rstrip('/')}/api/embed", json=payload, timeout=120)
        if resp.status_code == 404:
            resp = session.post(f"{base_url.rstrip('/')}/api/embeddings", json={"model": model, "prompt": text}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if "embeddings" in data:
            vectors.append(data["embeddings"][0])
        elif "embedding" in data:
            vectors.append(data["embedding"])
        else:
            raise RuntimeError(f"Unexpected Ollama embedding response keys: {list(data)}")
    return vectors


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def vector_scores(query: str, docs: list[dict[str, Any]], base_url: str, model: str) -> dict[str, float]:
    doc_texts = [doc["title"] + "\n" + doc["text"] for doc in docs]
    doc_vectors = ollama_embed(doc_texts, base_url, model)
    query_vector = ollama_embed([query], base_url, model)[0]
    return {doc["id"]: cosine(query_vector, vec) for doc, vec in zip(docs, doc_vectors)}


def char_ngrams(text: str, n: int = 2) -> list[str]:
    chars = [ch for ch in text.lower() if not ch.isspace()]
    if len(chars) < n:
        return chars
    return ["".join(chars[i : i + n]) for i in range(len(chars) - n + 1)]


def tfidf_ngram_scores(query: str, docs: list[dict[str, Any]], n: int = 2) -> dict[str, float]:
    doc_terms = [char_ngrams(doc["title"] + " " + doc["text"], n=n) for doc in docs]
    query_terms = char_ngrams(query, n=n)
    df = defaultdict(int)
    for terms in doc_terms:
        for term in set(terms):
            df[term] += 1
    total_docs = len(docs)

    def vectorize(terms: list[str]) -> dict[str, float]:
        counts = Counter(terms)
        vec = {}
        for term, count in counts.items():
            idf = math.log((total_docs + 1) / (df.get(term, 0) + 1)) + 1
            vec[term] = count * idf
        return vec

    query_vec = vectorize(query_terms)
    scores = {}
    for doc, terms in zip(docs, doc_terms):
        doc_vec = vectorize(terms)
        common = set(query_vec) & set(doc_vec)
        dot = sum(query_vec[t] * doc_vec[t] for t in common)
        q_norm = math.sqrt(sum(v * v for v in query_vec.values()))
        d_norm = math.sqrt(sum(v * v for v in doc_vec.values()))
        scores[doc["id"]] = dot / (q_norm * d_norm) if q_norm and d_norm else 0.0
    return scores


def rank_from_scores(scores: dict[str, float]) -> list[str]:
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def rrf(*rankings: list[str], k: int = 60) -> list[str]:
    scores = defaultdict(float)
    for ranking in rankings:
        for idx, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + idx + 1)
    return rank_from_scores(scores)


def evaluate_rankings(rows: list[dict[str, Any]], rankings: dict[str, list[str]], top_k: int) -> dict[str, float]:
    recalls = []
    reciprocal_ranks = []
    precisions = []
    for row in rows:
        relevant = set(row["relevant_doc_ids"])
        ranked = rankings[row["question"]]
        top = ranked[:top_k]
        hit_count = len(relevant & set(top))
        recalls.append(hit_count / len(relevant))
        precisions.append(hit_count / top_k)
        rr = 0.0
        for idx, doc_id in enumerate(ranked, start=1):
            if doc_id in relevant:
                rr = 1.0 / idx
                break
        reciprocal_ranks.append(rr)
    return {
        f"recall@{top_k}": sum(recalls) / len(recalls),
        f"precision@{top_k}": sum(precisions) / len(precisions),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }


def pct_change(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return (new - old) / old * 100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="atrag/evaluation/local_flow_eval/corpus.jsonl")
    parser.add_argument("--questions", default="atrag/evaluation/local_flow_eval/questions.csv")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--embedding-model", default="qwen3-embedding-0.6b")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-vector", action="store_true")
    parser.add_argument("--local-vectorizer", choices=["ollama", "char_tfidf"], default="ollama")
    parser.add_argument("--output", default="atrag/evaluation/local_flow_eval/report.json")
    args = parser.parse_args()

    corpus = load_corpus(Path(args.corpus))
    questions = load_questions(Path(args.questions))

    report: dict[str, Any] = {
        "corpus": args.corpus,
        "questions": args.questions,
        "top_k": args.top_k,
        "embedding_model": args.embedding_model,
        "flows": {},
        "improvements_percent": {
            "hybrid_vs_bm25_baseline": {},
            "hybrid_vs_vector_baseline": {},
        },
    }

    bm25_rankings = {}
    entity_rankings = {}
    vector_rankings = {}
    hybrid_rankings = {}

    start = time.perf_counter()
    for row in questions:
        q = row["question"]
        bm25 = bm25_scores(q, corpus)
        entity = entity_scores(q, corpus)
        bm25_rankings[q] = rank_from_scores(bm25)
        entity_rankings[q] = rank_from_scores(entity)
    report["flows"]["bm25_baseline"] = evaluate_rankings(questions, bm25_rankings, args.top_k)
    report["timing_seconds_bm25_entity"] = round(time.perf_counter() - start, 3)

    if not args.skip_vector:
        start = time.perf_counter()
        try:
            for row in questions:
                q = row["question"]
                if args.local_vectorizer == "char_tfidf":
                    vector_rankings[q] = rank_from_scores(tfidf_ngram_scores(q, corpus))
                else:
                    vector_rankings[q] = rank_from_scores(vector_scores(q, corpus, args.ollama_url, args.embedding_model))
                hybrid_rankings[q] = rrf(vector_rankings[q], bm25_rankings[q], entity_rankings[q])
            report["flows"]["vector_baseline"] = evaluate_rankings(questions, vector_rankings, args.top_k)
            report["flows"]["hybrid_vector_bm25_entity"] = evaluate_rankings(questions, hybrid_rankings, args.top_k)
            report["timing_seconds_vector_hybrid"] = round(time.perf_counter() - start, 3)
        except Exception as exc:
            report["vector_error"] = str(exc)

    if "hybrid_vector_bm25_entity" in report["flows"]:
        improved = report["flows"]["hybrid_vector_bm25_entity"]
        for baseline_name, output_name in [
            ("bm25_baseline", "hybrid_vs_bm25_baseline"),
            ("vector_baseline", "hybrid_vs_vector_baseline"),
        ]:
            if baseline_name not in report["flows"]:
                continue
            base = report["flows"][baseline_name]
            for key, value in improved.items():
                report["improvements_percent"][output_name][key] = pct_change(value, base.get(key, 0.0))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
