# 🔍 RAG

A Retrieval-Augmented Generation (RAG) system for answering questions about a fitness/gym company's internal policies, built from a Markdown document corpus.

## ⚙️ How It Works

The pipeline (`notebooks/pipeline.py`) runs a query through the following steps:

1. **Document Loading & Chunking** — Loads all `.md` files from the `corpus/` folder (excluding `README.md`) and splits each document into chunks using LangChain's `MarkdownHeaderTextSplitter`, split on `##` headers.
2. **Vector Database Setup** — Stores the chunks in a Chroma in-memory collection (`titanic-fitness-chunked`).
3. **Query Processing** — Encodes the user's query into an embedding using the `sentence-transformers` model `all-MiniLM-L6-v2`.
4. **Vector Search** — Retrieves the most similar chunk(s) from the Chroma collection via similarity search.
5. **Context Augmentation** — Builds a prompt combining the retrieved policy context with the user's question.
6. **Response Generation** — Sends the augmented prompt to OpenAI (`gpt-5.6-luna`) to generate the final answer.

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

Run the pipeline from the project root:

```bash
python -m notebooks.pipeline
```

This runs a set of example queries against the gym policy corpus and prints the generated answers.

## 📁 Project Structure

```
RAG/
├── corpus/           # Markdown documents (gym policies, by category)
├── notebooks/        # Pipeline and exploratory scripts
├── credentials.py    # OpenAI API key (not committed)
└── requirements.txt
```
