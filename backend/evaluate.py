"""
evaluate.py — RAG Evaluation Script
=====================================
Runs all 50 questions from eval_dataset.json against the live backend,
computes ROUGE-1, ROUGE-L, BLEU, token F1, exact match, and keyword
recall for both RAG and Memory answers, then saves a full report.

Usage:
    python evaluate.py                  # runs all 50 questions
    python evaluate.py --limit 10       # runs first 10 questions
    python evaluate.py --out report.json
"""

import re
import json
import math
import argparse
import time
from collections import Counter
from pathlib import Path
import urllib.request
import urllib.error

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_BASE   = "http://localhost:8000"
DATASET    = Path(__file__).parent / "eval_dataset.json"
OUT_REPORT = Path(__file__).parent / "eval_report.json"

# ─────────────────────────────────────────────────────────────────────────────
# METRIC FUNCTIONS (pure Python — no external ML libraries needed)
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def rouge1(hypothesis: str, reference: str) -> dict:
    """ROUGE-1: unigram overlap between hypothesis and reference."""
    hyp = tokenize(hypothesis)
    ref = tokenize(reference)
    if not hyp or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    hyp_c = Counter(hyp)
    ref_c = Counter(ref)
    overlap = sum((hyp_c & ref_c).values())
    precision = overlap / len(hyp)
    recall    = overlap / len(ref)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4)}


def lcs_length(a: list, b: list) -> int:
    """Length of Longest Common Subsequence."""
    m, n = len(a), len(b)
    # space-optimised DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev = curr
    return prev[n]


def rouge_l(hypothesis: str, reference: str) -> dict:
    """ROUGE-L: based on Longest Common Subsequence."""
    hyp = tokenize(hypothesis)
    ref = tokenize(reference)
    if not hyp or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = lcs_length(hyp, ref)
    precision = lcs / len(hyp)
    recall    = lcs / len(ref)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4)}


def ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))


def bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """Corpus BLEU up to max_n with brevity penalty."""
    hyp = tokenize(hypothesis)
    ref = tokenize(reference)
    if not hyp:
        return 0.0
    bp = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref)/len(hyp))
    precisions = []
    for n in range(1, max_n + 1):
        hyp_ng = ngrams(hyp, n)
        ref_ng = ngrams(ref, n)
        if not hyp_ng:
            precisions.append(0.0)
            continue
        overlap = sum((hyp_ng & ref_ng).values())
        precisions.append(overlap / sum(hyp_ng.values()))
    # geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0
    log_avg = sum(math.log(p) for p in precisions) / max_n
    return round(bp * math.exp(log_avg), 4)


def token_f1(hypothesis: str, reference: str) -> float:
    """Token-level F1 (used in QA benchmarks like SQuAD)."""
    hyp = tokenize(hypothesis)
    ref = tokenize(reference)
    if not hyp or not ref:
        return 0.0
    common = Counter(hyp) & Counter(ref)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(hyp)
    recall    = num_same / len(ref)
    return round(2 * precision * recall / (precision + recall), 4)


def exact_match(hypothesis: str, reference: str) -> int:
    """1 if answers match after normalisation, else 0."""
    def norm(s):
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9\s]", "", s)
        return " ".join(s.split())
    return int(norm(hypothesis) == norm(reference))


def keyword_recall(hypothesis: str, keywords: list[str]) -> float:
    """Fraction of expected keywords present in the hypothesis."""
    if not keywords:
        return 1.0
    hyp_lower = hypothesis.lower()
    hits = sum(1 for kw in keywords if kw.lower() in hyp_lower)
    return round(hits / len(keywords), 4)


def compute_all_metrics(hypothesis: str, reference: str, keywords: list[str]) -> dict:
    return {
        "rouge1":          rouge1(hypothesis, reference),
        "rouge_l":         rouge_l(hypothesis, reference),
        "bleu":            bleu(hypothesis, reference),
        "token_f1":        token_f1(hypothesis, reference),
        "exact_match":     exact_match(hypothesis, reference),
        "keyword_recall":  keyword_recall(hypothesis, keywords),
    }


# ─────────────────────────────────────────────────────────────────────────────
# API CALL
# ─────────────────────────────────────────────────────────────────────────────

def query_api(question: str) -> dict:
    payload = json.dumps({"question": question}).encode()
    req = urllib.request.Request(
        f"{API_BASE}/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(limit: int | None = None, out_path: Path = OUT_REPORT):
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    if limit:
        dataset = dataset[:limit]

    results   = []
    rag_totals = Counter()
    mem_totals = Counter()
    category_scores: dict[str, dict] = {}

    print(f"\n{'='*65}")
    print(f"  Nexus RAG Evaluator — {len(dataset)} questions")
    print(f"{'='*65}\n")

    for i, item in enumerate(dataset, 1):
        qid      = item["id"]
        question = item["question"]
        truth    = item["ground_truth"]
        keywords = item.get("expected_keywords", [])
        category = item.get("category", "Unknown")
        difficulty = item.get("difficulty", "medium")

        print(f"[{i:02d}/{len(dataset)}] {qid} | {difficulty:6s} | {question[:60]}...")

        try:
            resp = query_api(question)
            rag_ans = resp.get("rag_answer", "")
            mem_ans = resp.get("memory_answer", "")
            sources = resp.get("sources", [])
        except Exception as e:
            print(f"  ⚠  API error: {e}")
            rag_ans = mem_ans = ""
            sources = []

        rag_metrics = compute_all_metrics(rag_ans, truth, keywords)
        mem_metrics = compute_all_metrics(mem_ans, truth, keywords)

        # accumulate totals
        for k, v in rag_metrics.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    rag_totals[f"{k}_{sk}"] += sv
            else:
                rag_totals[k] += v

        for k, v in mem_metrics.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    mem_totals[f"{k}_{sk}"] += sv
            else:
                mem_totals[k] += v

        # per-category
        if category not in category_scores:
            category_scores[category] = {"rag": Counter(), "mem": Counter(), "count": 0}
        category_scores[category]["count"] += 1
        for k, v in rag_metrics.items():
            val = v["f1"] if isinstance(v, dict) else v
            category_scores[category]["rag"][k] += val
        for k, v in mem_metrics.items():
            val = v["f1"] if isinstance(v, dict) else v
            category_scores[category]["mem"][k] += val

        results.append({
            "id":          qid,
            "category":    category,
            "difficulty":  difficulty,
            "question":    question,
            "ground_truth":truth,
            "keywords":    keywords,
            "rag_answer":  rag_ans,
            "mem_answer":  mem_ans,
            "sources":     sources,
            "rag_metrics": rag_metrics,
            "mem_metrics": mem_metrics,
        })

        # brief inline summary
        r1_rag = rag_metrics["rouge1"]["f1"]
        r1_mem = mem_metrics["rouge1"]["f1"]
        kw_rag = rag_metrics["keyword_recall"]
        kw_mem = mem_metrics["keyword_recall"]
        print(f"  ROUGE-1  RAG={r1_rag:.3f}  MEM={r1_mem:.3f}  |  "
              f"KW-Recall  RAG={kw_rag:.3f}  MEM={kw_mem:.3f}")

        time.sleep(0.5)   # be polite to the API

    n = len(dataset)

    def avg(counter, key):
        return round(counter[key] / n, 4)

    # ── AGGREGATE SUMMARY ─────────────────────────────────────────────────────
    summary = {
        "total_questions": n,
        "rag": {
            "rouge1_f1":       avg(rag_totals, "rouge1_f1"),
            "rouge1_precision":avg(rag_totals, "rouge1_precision"),
            "rouge1_recall":   avg(rag_totals, "rouge1_recall"),
            "rouge_l_f1":      avg(rag_totals, "rouge_l_f1"),
            "bleu":            avg(rag_totals, "bleu"),
            "token_f1":        avg(rag_totals, "token_f1"),
            "exact_match":     avg(rag_totals, "exact_match"),
            "keyword_recall":  avg(rag_totals, "keyword_recall"),
        },
        "memory": {
            "rouge1_f1":       avg(mem_totals, "rouge1_f1"),
            "rouge1_precision":avg(mem_totals, "rouge1_precision"),
            "rouge1_recall":   avg(mem_totals, "rouge1_recall"),
            "rouge_l_f1":      avg(mem_totals, "rouge_l_f1"),
            "bleu":            avg(mem_totals, "bleu"),
            "token_f1":        avg(mem_totals, "token_f1"),
            "exact_match":     avg(mem_totals, "exact_match"),
            "keyword_recall":  avg(mem_totals, "keyword_recall"),
        },
    }

    # ── CATEGORY BREAKDOWN ────────────────────────────────────────────────────
    category_summary = {}
    for cat, data in category_scores.items():
        cnt = data["count"]
        category_summary[cat] = {
            "count": cnt,
            "rag": {k: round(v/cnt, 4) for k, v in data["rag"].items()},
            "mem": {k: round(v/cnt, 4) for k, v in data["mem"].items()},
        }

    report = {
        "summary":           summary,
        "category_breakdown":category_summary,
        "per_question":      results,
    }

    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── PRINT FINAL TABLE ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  EVALUATION RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Metric':<22} {'RAG':>10} {'MEMORY':>10}  {'Winner':>8}")
    print(f"  {'-'*58}")

    metrics_display = [
        ("ROUGE-1 F1",       "rouge1_f1"),
        ("ROUGE-1 Precision","rouge1_precision"),
        ("ROUGE-1 Recall",   "rouge1_recall"),
        ("ROUGE-L F1",       "rouge_l_f1"),
        ("BLEU",             "bleu"),
        ("Token F1",         "token_f1"),
        ("Exact Match",      "exact_match"),
        ("Keyword Recall",   "keyword_recall"),
    ]
    for label, key in metrics_display:
        r = summary["rag"][key]
        m = summary["memory"][key]
        winner = "RAG ✓" if r >= m else "MEM ✓"
        print(f"  {label:<22} {r:>10.4f} {m:>10.4f}  {winner:>8}")

    print(f"\n  Full report saved → {out_path}\n")
    return report


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG vs Memory answers")
    parser.add_argument("--limit", type=int, default=None, help="Max questions to evaluate")
    parser.add_argument("--out",   type=str,  default=str(OUT_REPORT), help="Output JSON path")
    args = parser.parse_args()
    evaluate(limit=args.limit, out_path=Path(args.out))
