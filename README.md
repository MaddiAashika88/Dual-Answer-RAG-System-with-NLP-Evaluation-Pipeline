# 🌌 Nexus AI — Dual-Answer RAG System

A **Retrieval-Augmented Generation (RAG)** API that answers every question *twice* — once grounded in your document knowledge base, and once from the model's pre-trained memory — then scores both answers side-by-side using a full suite of NLP metrics.

Built with **FastAPI**, **ChromaDB**, **Groq (LLaMA 3.3 70B)**, and a clean dark-themed frontend.

---

## ✨ Features

- **Dual-answer architecture** — every query returns a RAG answer (document-grounded) and a Memory answer (model knowledge only), letting you compare retrieval quality at a glance
- **Live NLP metrics** — ROUGE-1, ROUGE-L, BLEU, Token F1, Exact Match, Keyword Recall, and a composite Accuracy score computed on every response
- **ChromaDB vector store** — documents are embedded and indexed on startup; new documents can be added at runtime via `/add_document`
- **50-question eval dataset** — run a full offline evaluation via `evaluate.py` or trigger it through the `/evaluate` endpoint
- **Context-as-reference fallback** — for questions outside the eval dataset, retrieved context is used as the scoring reference so metrics are always meaningful
- **Category & difficulty breakdowns** — evaluation reports segment results by topic category and question difficulty
- **Chat history sidebar** — the frontend maintains a session history with per-turn metric cards
- **Zero ML-library dependencies** — all metrics are implemented in pure Python (no `nltk`, `rouge_score`, or `sacrebleu` required)

---

## 🗂 Project Structure

```
.
├── main.py            # FastAPI backend — routes, RAG pipeline, metrics engine
├── index.html         # Single-file dark-themed chat frontend
├── documents.json     # Knowledge base (Space & Science documents)
├── eval_dataset.json  # 50 ground-truth Q&A pairs for evaluation
├── evaluate.py        # Standalone CLI evaluation script
└── requirements.txt   # Python dependencies
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Groq API key

Open `main.py` and replace the placeholder:

```python
GROQ_API_KEY = "your-groq-api-key-here"
```

Or export it as an environment variable and update the line to:

```python
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Start the server

```bash
uvicorn main:app --reload --port 8000
```

### 4. Open the frontend

Open `index.html` directly in your browser (no build step needed). The frontend talks to `http://localhost:8000` by default.

---

## 🔌 API Reference

### `POST /query`

Submit a question and receive dual answers with metrics.

**Request**
```json
{
  "question": "What did the James Webb Space Telescope discover in 2023?",
  "chat_id": "optional-session-id"
}
```

**Response**
```json
{
  "rag_answer": "...",
  "memory_answer": "...",
  "sources": ["source-a", "source-b"],
  "chat_id": "abc-123",
  "metrics": {
    "mode": "dataset_match",
    "rag": {
      "accuracy": 0.74,
      "rouge1": { "precision": 0.81, "recall": 0.68, "f1": 0.74 },
      "rouge_l": { "f1": 0.71 },
      "bleu": 0.52,
      "token_f1": 0.74,
      "exact_match": 0.0,
      "keyword_recall": 0.90,
      "context_relevance": 0.88
    },
    "memory": { "..." : "..." }
  }
}
```

---

### `POST /evaluate`

Run batch evaluation against the built-in eval dataset.

**Request**
```json
{
  "limit": 10,
  "question_ids": ["q001", "q002"]
}
```

Returns per-question metrics plus aggregate summaries by category and difficulty. Results are also saved to `eval_report.json`.

---

### `POST /add_document`

Add a new document to the ChromaDB index at runtime.

```json
{
  "text": "NASA's Artemis III mission ...",
  "source": "nasa.gov",
  "title": "Artemis III Overview",
  "category": "Space Exploration"
}
```

---

### `GET /health`

```json
{ "status": "ok", "docs_indexed": 42, "eval_questions": 50 }
```

### `GET /documents` — list all indexed documents  
### `GET /eval_dataset` — list all evaluation questions

---

## 📊 Offline Evaluation (CLI)

Run `evaluate.py` against a live backend to generate a full report:

```bash
# Evaluate all 50 questions
python evaluate.py

# Evaluate first 10 questions only
python evaluate.py --limit 10

# Save report to a custom path
python evaluate.py --out results/my_report.json
```

The script prints a formatted summary table and saves the full per-question breakdown to `eval_report.json`:

```
═════════════════════════════════════════════════════════════════
  EVALUATION RESULTS SUMMARY
═════════════════════════════════════════════════════════════════
  Metric                       RAG     MEMORY    Winner
  ─────────────────────────────────────────────────────────
  ROUGE-1 F1              0.7412     0.4821    RAG ✓
  ROUGE-1 Precision       0.8100     0.5200    RAG ✓
  ROUGE-1 Recall          0.6900     0.4500    RAG ✓
  ROUGE-L F1              0.7050     0.4600    RAG ✓
  BLEU                    0.5230     0.2900    RAG ✓
  Token F1                0.7400     0.4800    RAG ✓
  Exact Match             0.0600     0.0200    RAG ✓
  Keyword Recall          0.8900     0.6100    RAG ✓
```

---

## 📐 Metrics Explained

| Metric | Description |
|---|---|
| **Accuracy** | Composite of ROUGE-1 F1 + Token F1 + Keyword Recall (0–1) |
| **ROUGE-1** | Unigram precision / recall / F1 between answer and ground truth |
| **ROUGE-L** | Longest Common Subsequence-based overlap |
| **BLEU** | n-gram precision with brevity penalty (up to 4-gram) |
| **Token F1** | SQuAD-style token overlap between answer and reference |
| **Exact Match** | 1 if normalized answer equals normalized ground truth |
| **Keyword Recall** | Fraction of domain-specific expected keywords present in the answer |
| **Context Relevance** | Fraction of question keywords covered by retrieved context |

All metrics are implemented in **pure Python** with no external NLP libraries.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Backend | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| LLM | [Groq](https://groq.com/) — LLaMA 3.3 70B Versatile |
| Vector Store | [ChromaDB](https://www.trychroma.com/) with default embedding function |
| Frontend | Vanilla HTML/CSS/JS (no framework, no build step) |
| Metrics | Pure Python (stdlib only) |

---

## ⚙️ Configuration

All config lives at the top of `main.py`:

```python
GROQ_API_KEY = "..."          # Your Groq API key
GROQ_MODEL   = "llama-3.3-70b-versatile"  # Model to use
```

The ChromaDB collection is in-memory by default (`chromadb.Client()`). To persist data across restarts, swap it for a persistent client:

```python
chroma_client = chromadb.PersistentClient(path="./chroma_db")
```

---

## 📄 License

MIT
