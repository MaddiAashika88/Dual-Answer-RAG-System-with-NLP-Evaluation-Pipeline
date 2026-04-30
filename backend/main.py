import os
import re
import math
import json
import uuid
from collections import Counter
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import groq as groq_module

# ─── CONFIG ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = "Your Groq API Key"
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ─── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Dual-Answer RAG API with Evaluation")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── GROQ CLIENT ───────────────────────────────────────────────────────────────
client = groq_module.Groq(api_key=GROQ_API_KEY)

# ─── CHROMA ────────────────────────────────────────────────────────────────────
chroma_client = chromadb.Client()
ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(name="rag_docs", embedding_function=ef)

# ─── LOAD DOCUMENTS ────────────────────────────────────────────────────────────
_docs_path = Path(__file__).parent / "documents.json"
with open(_docs_path, "r", encoding="utf-8") as _f:
    TRAINING_DOCS = json.load(_f)
print(f"📄 Loaded {len(TRAINING_DOCS)} documents from documents.json")

if collection.count() == 0:
    collection.add(
        documents=[d["text"] for d in TRAINING_DOCS],
        ids=[d["id"] for d in TRAINING_DOCS],
        metadatas=[{"source": d["source"], "title": d.get("title",""), "category": d.get("category","")}
                   for d in TRAINING_DOCS],
    )
    print(f"✅ Seeded {len(TRAINING_DOCS)} documents into ChromaDB")

# ─── LOAD EVAL DATASET ─────────────────────────────────────────────────────────
_eval_path = Path(__file__).parent / "eval_dataset.json"
EVAL_DATASET = json.loads(_eval_path.read_text(encoding="utf-8")) if _eval_path.exists() else []
# Build a quick lookup by normalised question
EVAL_LOOKUP = {q["question"].strip().lower(): q for q in EVAL_DATASET}
print(f"📊 Loaded {len(EVAL_DATASET)} evaluation questions")

# ══════════════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS  (pure Python — zero extra dependencies)
# ══════════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> list:
    return [t for t in re.sub(r"[^a-z0-9\s]", " ", text.lower()).split() if t]

# ── ROUGE-1 ────────────────────────────────────────────────────────────────────
def rouge1(hyp: str, ref: str) -> dict:
    h, r = tokenize(hyp), tokenize(ref)
    if not h or not r:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    overlap = sum((Counter(h) & Counter(r)).values())
    p  = overlap / len(h)
    rc = overlap / len(r)
    f1 = 2*p*rc/(p+rc) if (p+rc) else 0.0
    return {"precision": round(p,4), "recall": round(rc,4), "f1": round(f1,4)}

# ── ROUGE-L ────────────────────────────────────────────────────────────────────
def _lcs(a, b):
    prev = [0]*(len(b)+1)
    for x in a:
        curr = [0]*(len(b)+1)
        for j, y in enumerate(b, 1):
            curr[j] = prev[j-1]+1 if x==y else max(curr[j-1], prev[j])
        prev = curr
    return prev[len(b)]

def rouge_l(hyp: str, ref: str) -> dict:
    h, r = tokenize(hyp), tokenize(ref)
    if not h or not r:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = _lcs(h, r)
    p  = lcs/len(h); rc = lcs/len(r)
    f1 = 2*p*rc/(p+rc) if (p+rc) else 0.0
    return {"precision": round(p,4), "recall": round(rc,4), "f1": round(f1,4)}

# ── BLEU ───────────────────────────────────────────────────────────────────────
def bleu(hyp: str, ref: str, max_n: int = 4) -> float:
    h, r = tokenize(hyp), tokenize(ref)
    if not h: return 0.0
    bp = 1.0 if len(h) >= len(r) else math.exp(1 - len(r)/len(h))
    precs = []
    for n in range(1, max_n+1):
        hng = Counter(tuple(h[i:i+n]) for i in range(len(h)-n+1))
        rng = Counter(tuple(r[i:i+n]) for i in range(len(r)-n+1))
        if not hng: precs.append(0.0); continue
        precs.append(sum((hng & rng).values()) / sum(hng.values()))
    if any(p==0 for p in precs): return 0.0
    return round(bp * math.exp(sum(math.log(p) for p in precs)/max_n), 4)

# ── TOKEN F1 ───────────────────────────────────────────────────────────────────
def token_f1(hyp: str, ref: str) -> float:
    h, r = tokenize(hyp), tokenize(ref)
    if not h or not r: return 0.0
    common = sum((Counter(h) & Counter(r)).values())
    if not common: return 0.0
    p = common/len(h); rc = common/len(r)
    return round(2*p*rc/(p+rc), 4)

# ── EXACT MATCH ────────────────────────────────────────────────────────────────
def exact_match(hyp: str, ref: str) -> float:
    def norm(s): return " ".join(re.sub(r"[^a-z0-9\s]","",s.lower().strip()).split())
    return 1.0 if norm(hyp)==norm(ref) else 0.0

# ── KEYWORD RECALL ─────────────────────────────────────────────────────────────
def keyword_recall(hyp: str, keywords: list) -> float:
    if not keywords: return 1.0
    hyp_l = hyp.lower()
    return round(sum(1 for kw in keywords if kw.lower() in hyp_l) / len(keywords), 4)

# ── CONTEXT RELEVANCE (how well retrieved docs cover the question) ─────────────
def context_relevance(question: str, context: str) -> float:
    """Fraction of question keywords found in the retrieved context."""
    q_tokens = set(tokenize(question))
    c_tokens  = set(tokenize(context))
    stop = {"the","a","an","is","are","was","were","of","in","on","at","to","for",
            "and","or","but","not","it","its","this","that","with","by","as","be"}
    q_tokens -= stop
    if not q_tokens: return 1.0
    return round(len(q_tokens & c_tokens) / len(q_tokens), 4)

# ── ACCURACY (keyword-recall-based, 0-100%) ────────────────────────────────────
def accuracy(hyp: str, ref: str, keywords: list) -> float:
    """
    Composite accuracy = average of:
      - ROUGE-1 F1    (word overlap)
      - Token F1      (QA-style overlap)
      - Keyword Recall (domain-specific facts)
    Expressed as a 0-1 float.
    """
    r1  = rouge1(hyp, ref)["f1"]
    tf1 = token_f1(hyp, ref)
    kwr = keyword_recall(hyp, keywords)
    return round((r1 + tf1 + kwr) / 3, 4)

# ── ALL METRICS ────────────────────────────────────────────────────────────────
def all_metrics(hyp: str, ref: str, keywords: list, question: str = "", context: str = "") -> dict:
    acc = accuracy(hyp, ref, keywords)
    return {
        "accuracy":        acc,                          # ← composite 0-1
        "rouge1":          rouge1(hyp, ref),
        "rouge_l":         rouge_l(hyp, ref),
        "bleu":            bleu(hyp, ref),
        "token_f1":        token_f1(hyp, ref),
        "exact_match":     exact_match(hyp, ref),
        "keyword_recall":  keyword_recall(hyp, keywords),
        "context_relevance": context_relevance(question, context) if context else None,
    }

# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    question: str
    chat_id: Optional[str] = None

class DualAnswer(BaseModel):
    rag_answer: str
    memory_answer: str
    sources: List[str]
    chat_id: str
    metrics: dict          # always present now

class DocRequest(BaseModel):
    text: str
    source: str
    title: Optional[str] = ""
    category: Optional[str] = ""
    id: Optional[str] = None

class EvalRequest(BaseModel):
    limit: Optional[int] = None
    question_ids: Optional[List[str]] = None

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_context(question: str, n_results: int = 4):
    results = collection.query(query_texts=[question], n_results=n_results)
    docs    = results["documents"][0] if results["documents"] else []
    metas   = results["metadatas"][0]  if results["metadatas"]  else []
    sources = [m.get("source","Unknown") for m in metas]
    context = "\n\n".join(f"[Source: {s}]\n{d}" for d, s in zip(docs, sources))
    return context, sources

def ask_groq(system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user",  "content":user_prompt}],
        temperature=0.6,
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()

def _aggregate_summary(results: list, n: int) -> dict:
    rag_acc, mem_acc = Counter(), Counter()
    for r in results:
        for k, v in r["rag_metrics"].items():
            if v is None: continue
            val = v["f1"] if isinstance(v, dict) else v
            rag_acc[k] += val
        for k, v in r["mem_metrics"].items():
            if v is None: continue
            val = v["f1"] if isinstance(v, dict) else v
            mem_acc[k] += val

    def avg(c): return {k: round(v/n, 4) for k, v in c.items()}

    cats: dict = {}
    for r in results:
        cat = r["category"]
        if cat not in cats:
            cats[cat] = {"count": 0, "rag": Counter(), "mem": Counter()}
        cats[cat]["count"] += 1
        for k, v in r["rag_metrics"].items():
            if v is None: continue
            cats[cat]["rag"][k] += v["f1"] if isinstance(v, dict) else v
        for k, v in r["mem_metrics"].items():
            if v is None: continue
            cats[cat]["mem"][k] += v["f1"] if isinstance(v, dict) else v

    cat_summary = {cat: {"count": d["count"],
                          "rag": {k: round(v/d["count"],4) for k,v in d["rag"].items()},
                          "mem": {k: round(v/d["count"],4) for k,v in d["mem"].items()}}
                   for cat, d in cats.items()}

    diffs: dict = {}
    for r in results:
        diff = r.get("difficulty","medium")
        if diff not in diffs:
            diffs[diff] = {"count": 0, "rag": Counter(), "mem": Counter()}
        diffs[diff]["count"] += 1
        for k, v in r["rag_metrics"].items():
            if v is None: continue
            diffs[diff]["rag"][k] += v["f1"] if isinstance(v, dict) else v
        for k, v in r["mem_metrics"].items():
            if v is None: continue
            diffs[diff]["mem"][k] += v["f1"] if isinstance(v, dict) else v

    diff_summary = {d: {"count": data["count"],
                         "rag": {k: round(v/data["count"],4) for k,v in data["rag"].items()},
                         "mem": {k: round(v/data["count"],4) for k,v in data["mem"].items()}}
                    for d, data in diffs.items()}

    return {
        "total_questions": n,
        "overall":         {"rag": avg(rag_acc), "memory": avg(mem_acc)},
        "by_category":     cat_summary,
        "by_difficulty":   diff_summary,
    }

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status":"ok","docs_indexed":collection.count(),"eval_questions":len(EVAL_DATASET)}

@app.get("/documents")
def list_documents():
    return {"total": len(TRAINING_DOCS), "documents": TRAINING_DOCS}

@app.get("/eval_dataset")
def get_eval_dataset():
    return {"total": len(EVAL_DATASET), "questions": EVAL_DATASET}


@app.post("/query", response_model=DualAnswer)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    chat_id = req.chat_id or str(uuid.uuid4())
    context, sources = retrieve_context(req.question)

    rag_system = (
        "You are an expert Space & Science assistant. You have been given context documents "
        "covering real post-2023 discoveries, missions, and breakthroughs. "
        "Answer using ONLY the provided context. Be concise, factual, cite the source."
    )
    rag_answer = ask_groq(rag_system, f"Context:\n{context}\n\nQuestion: {req.question}")

    memory_system = (
        "You are an expert Space & Science assistant answering based solely on your trained "
        "knowledge up to early 2023. Do NOT mention anything after that date. "
        "Acknowledge if your knowledge may be limited. Be concise and accurate."
    )
    memory_answer = ask_groq(memory_system, req.question)

    # ── Always compute metrics ─────────────────────────────────────────────────
    # DATASET MATCH → use real ground truth + expected keywords (most accurate)
    # NEW QUESTION  → use the retrieved context as the reference.
    #                 The context IS the authoritative source the RAG should follow,
    #                 so scoring both answers against it is fair:
    #                 RAG is rewarded for faithfully using the docs,
    #                 Memory is penalised for diverging from them.
    #                 Auto-extract keywords = all notable tokens (numbers, proper nouns,
    #                 dates) found in the context but NOT common stop-words.
    match = EVAL_LOOKUP.get(req.question.strip().lower())

    if match:
        ref      = match["ground_truth"]
        keywords = match.get("expected_keywords", [])
        mode     = "dataset_match"
    else:
        # Use raw context text (without [Source:…] headers) as the reference
        raw_context_texts = [d for d in context.split("\n\n") if not d.startswith("[Source:")]
        ref = " ".join(raw_context_texts).strip() or context

        # Auto-extract meaningful keywords from context:
        # keep tokens that are ≥4 chars, not stop-words, appear in context
        STOP = {
            "this","that","with","from","have","been","they","their","which",
            "will","were","also","into","than","more","some","when","then",
            "about","after","over","such","only","each","both","very","most",
            "other","many","much","make","like","just","well","even","used",
            "using","based","made","known","first","last","next","the","and",
            "for","are","was","its","not","but","has","had","can","all","any",
            "one","two","three","four","five","six","seven","eight","nine","ten",
            "new","high","low","large","small","long","short","early","late",
        }
        ctx_tokens = re.findall(r'\b[A-Za-z0-9][A-Za-z0-9\-]{3,}\b', context)
        seen, keywords = set(), []
        for tok in ctx_tokens:
            tl = tok.lower()
            if tl not in STOP and tl not in seen:
                seen.add(tl)
                keywords.append(tok)
        keywords = keywords[:20]   # cap at 20 to keep scoring meaningful
        mode     = "context_as_reference"

    rag_m = all_metrics(rag_answer,    ref, keywords, req.question, context)
    mem_m = all_metrics(memory_answer, ref, keywords, req.question, context)

    metrics_payload = {
        "mode":          mode,
        "ground_truth":  ref if match else None,
        "context_ref":   None if match else ref[:400]+"…" if len(ref)>400 else ref,
        "keywords":      keywords,
        "rag":           rag_m,
        "memory":        mem_m,
    }

    return DualAnswer(
        rag_answer=rag_answer,
        memory_answer=memory_answer,
        sources=list(dict.fromkeys(sources)),
        chat_id=chat_id,
        metrics=metrics_payload,
    )


@app.post("/evaluate")
def run_evaluation(req: EvalRequest):
    dataset = EVAL_DATASET
    if req.question_ids:
        dataset = [q for q in dataset if q["id"] in req.question_ids]
    if req.limit:
        dataset = dataset[:req.limit]
    if not dataset:
        raise HTTPException(status_code=404, detail="No matching questions found")

    per_q = []
    for item in dataset:
        question = item["question"]
        truth    = item["ground_truth"]
        keywords = item.get("expected_keywords", [])

        try:
            context, sources = retrieve_context(question)
            rag_ans = ask_groq(
                "You are an expert Space & Science assistant. Answer using ONLY the provided context. Be concise and factual.",
                f"Context:\n{context}\n\nQuestion: {question}"
            )
            mem_ans = ask_groq(
                "You are an expert Space & Science assistant answering only from knowledge up to early 2023. Acknowledge if outdated. Be concise.",
                question
            )
        except Exception as e:
            rag_ans = mem_ans = f"[ERROR: {e}]"
            sources = context = ""

        rag_m = all_metrics(rag_ans, truth, keywords, question, context)
        mem_m = all_metrics(mem_ans, truth, keywords, question, "")

        per_q.append({
            "id": item["id"], "category": item.get("category",""),
            "difficulty": item.get("difficulty","medium"),
            "question": question, "ground_truth": truth, "keywords": keywords,
            "rag_answer": rag_ans, "mem_answer": mem_ans, "sources": sources,
            "rag_metrics": rag_m, "mem_metrics": mem_m,
        })

    summary = _aggregate_summary(per_q, len(per_q))
    report_path = Path(__file__).parent / "eval_report.json"
    report_path.write_text(json.dumps({"summary": summary, "per_question": per_q}, indent=2), encoding="utf-8")
    return {"summary": summary, "per_question": per_q}


@app.post("/add_document")
def add_document(doc: DocRequest):
    doc_id = doc.id or str(uuid.uuid4())
    collection.add(
        documents=[doc.text], ids=[doc_id],
        metadatas=[{"source": doc.source, "title": doc.title, "category": doc.category}],
    )
    return {"status": "added", "id": doc_id, "total": collection.count()}
