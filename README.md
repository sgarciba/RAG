# 🔍 RAG — Gym Policy Q&A

A Retrieval-Augmented Generation (RAG) system that answers questions about a fitness/gym company's internal policies, grounded in a Markdown document corpus — plus an evaluation framework that compares chunking strategies to find the best one for this use case.

## 🎯 Problem Statement & Goal

Gym staff and members need fast, accurate answers to policy questions (cancellations, guest passes, safety procedures, membership tiers...) that live scattered across many long-form Markdown documents. Reading through every policy file to find the right clause is slow, and copy-pasting the wrong document into an LLM risks confidently wrong answers.

**Goal:** build a RAG pipeline that retrieves the right policy snippet and generates a grounded answer — and, since chunking strategy is one of the biggest levers on RAG quality, **systematically evaluate four different chunking methods** to decide which one best balances retrieval accuracy and answer faithfulness for this use case.

## ⚙️ How It Works

The pipeline (`pipeline.py`) runs a query through the following steps:

1. **Document Loading & Chunking** — Loads all `.md` files from the `corpus/` folder (excluding `README.md`) and splits each document into chunks using one of four configurable chunking methods (see `config.py`).
2. **Vector Database Setup** — Embeds and stores the chunks in a Chroma in-memory collection, one per chunking method.
3. **Query Processing** — Encodes the user's query into an embedding using the `sentence-transformers` model `all-MiniLM-L6-v2`.
4. **Vector Search** — Retrieves the most similar chunk(s) from the Chroma collection via similarity search.
5. **Context Augmentation** — Builds a prompt combining the retrieved policy context with the user's question.
6. **Response Generation** — Sends the augmented prompt to OpenAI (`gpt-5.6-luna`) to generate the final answer.

### Chunking methods compared

| Method | Strategy |
|---|---|
| `markdown_header` | Splits on `##` headers, preserving each document's natural structure |
| `rec_char_256_0` | Fixed-size chunks of 256 characters, no overlap |
| `rec_char_512_50` | Fixed-size chunks of 512 characters, 50-character overlap |
| `rec_char_1024_100` | Fixed-size chunks of 1024 characters, 100-character overlap |

## 🛠️ Setup

```bash
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your OpenAI API key in a `credentials.py` file at the project root:

```python
OPENAI_API_KEY = "your-api-key-here"
```

## 🚀 Usage

All commands are run from the project root.

**Run the pipeline** (example queries against the gym policy corpus, printed answers):

```bash
python pipeline.py
```

**Run the evaluation** (regenerates results for all four chunking methods — calls the OpenAI API, so it costs credits):

```bash
# 1. Retrieval: runs every eval question through each chunking method, saves retrieval_results.parquet
python -m evaluation.eval_pipeline

# 2. Faithfulness: generates an answer per question/method and LLM-judges how grounded it is, saves faithfulness_results.parquet
python -m evaluation.faithfulness

# 3. Performance: combines both result files into the Recall@2 / MRR@2 / Faithfulness summary table
python -m evaluation.performance
```

## 📁 Project Structure

```
RAG/
├── corpus/                              # Markdown policy documents, by category
├── evaluation/
│   ├── eval_dataset.json                # 36 labeled Q&A pairs used for evaluation
│   ├── eval_pipeline.py                 # Runs retrieval for every chunking method
│   ├── faithfulness.py                  # Generates answers and LLM-judges groundedness
│   ├── performance.py                   # Computes Recall@2, MRR@2, Faithfulness
│   ├── retrieval_results.parquet        # Saved retrieval results
│   └── faithfulness_results.parquet     # Saved faithfulness results
├── pipeline.py                          # Core RAG pipeline
├── config.py                            # Chunking method definitions
├── credentials.py                       # OpenAI API key (not committed)
└── requirements.txt
```

## 📊 Insights

Each chunking method was evaluated on 36 policy questions across 16 documents, using three metrics:

- **Recall@2** — did the correct source document appear in the top 2 retrieved chunks?
- **MRR@2** — how high up the ranking was the correct document (1st place beats 2nd)?
- **Faithfulness** — of the claims in the generated answer, what fraction are actually supported by the retrieved context (i.e., not hallucinated)?

| Chunking method | Recall@2 | MRR@2 | Faithfulness |
|---|---|---|---|
| `markdown_header` | 0.972 | **0.958** | **0.979** |
| `rec_char_256_0` | **1.000** | 0.972 | 0.902 |
| `rec_char_512_50` | **1.000** | 0.972 | 0.929 |
| `rec_char_1024_100` | 0.972 | 0.903 | 0.938 |

### 🏆 Best choice for this use case: `markdown_header`

The fixed-size splitters technically win on raw retrieval (perfect Recall@2), but `markdown_header` gives up only a single retrieval miss out of 36 questions while producing by far the most faithful answers — a ~5–8 point jump in faithfulness over every other method. For a policy assistant, an ungrounded answer is far more costly than an occasional missed retrieval, so structure-aware chunking is the better trade-off here.

**Why it wins:** the policy documents are already organized under `##` headers, so splitting on structure keeps each chunk as one complete, self-contained policy rule. Fixed-size splitting cuts through that structure — sometimes mid-sentence or mid-rule — handing the model partial context it then has to guess how to complete, which shows up directly as hallucination.

### 💡 Interesting insights

- **Retrieval accuracy and answer faithfulness aren't the same thing.** All four methods retrieve well (≥97% recall), but faithfulness spans a much wider range (0.90–0.98). Optimizing chunking purely for retrieval metrics would have picked the *worst*-performing method for groundedness.
- **Bigger chunks aren't automatically better.** `rec_char_1024_100` has the largest chunk size but the *lowest* MRR@2 of all four methods — larger chunks dilute the similarity signal, pulling in more irrelevant surrounding text alongside the relevant part.

### 🔭 What I'd improve next

1. **Hybrid retrieval + reranking.** Combine dense embedding search with keyword-based search (e.g. BM25) and add a reranking step on top of the top-k candidates — this typically improves recall further and is a standard production RAG upgrade.
2. **Deeper evaluation.** Scale the eval set beyond 36 questions, add adversarial/no-answer questions to test refusal behavior, and validate the LLM-as-judge faithfulness score against a small human-labeled sample to check it isn't systematically biased.
