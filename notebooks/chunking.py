from langchain.text_splitter import MarkdownHeaderTextSplitter
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

corpus_dir = Path("corpus")
doc_paths = [str(p) for p in sorted(corpus_dir.rglob("*.md"))][1:]
docs = [Path(p).read_text(encoding="utf-8") for p in doc_paths]
doc_ids = [f"doc_{i+1}" for i in range(len(docs))]


client = chromadb.Client()
model = SentenceTransformer('all-MiniLM-L6-v2')



## 1. Collection chunked
collection_chunked = client.create_collection("titanic-fitness-chunked")

headers_to_split_on = [
    ("##", "Header 2"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)

# Split the documents into chunks
chunks = []
for doc in docs:
    doc_chunk = splitter.split_text(doc)
    page_contents = [doc.page_content for doc in doc_chunk]
    chunks.extend(page_contents)
    

chunk_ids = [f"chunk_{i+1}" for i in range(len(chunks))]

print(f"There is a total of {len(chunks)} chunks across {len(docs)} documents")

embeddings_chunked = model.encode(chunks)
collection_chunked.add(
    documents=chunks,
    embeddings=embeddings_chunked.tolist(),
    ids=chunk_ids
)

    
## 2. Collection No Chunking
collection_no_chunking = client.create_collection("titanic-fitness")

embeddings_no_chunking = model.encode(docs)
collection_no_chunking.add(
    documents=docs,
    embeddings=embeddings_no_chunking.tolist(),
    ids=doc_ids
)


## 3. Evaluation

# Test Vector Search
query = "What are the main facilities of the gym?"

results_chunked = collection_chunked.query(
    query_texts=[query],
    n_results=1
)

results_no_chunking = collection_no_chunking.query(
    query_texts=[query],
    n_results=1
)


print(f"Query: '{query}'")

similarity_no_chunking = 1 - results_no_chunking['distances'][0][0]
similarity_chunked = 1 - results_chunked['distances'][0][0]

print(f" Results No Chunking: Similarity: {similarity_no_chunking:.3f} - {results_no_chunking['documents'][0]}")
print(f" Results Chunked: Similarity: {similarity_chunked:.3f} - {results_chunked['documents'][0]}")


