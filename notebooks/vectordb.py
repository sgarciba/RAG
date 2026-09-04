import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

corpus_dir = Path("corpus")
doc_paths = [str(p) for p in sorted(corpus_dir.rglob("*.md"))][1:]
docs = [Path(p).read_text(encoding="utf-8") for p in doc_paths]
doc_ids = [f"doc_{i+1}" for i in range(len(docs))]


client = chromadb.Client()
collection = client.create_collection("titanic-fitness")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs)
print(f"Model loaded: {model.get_sentence_embedding_dimension()} dimensions")

## Use a sample document to test VectorDB
#test_doc = docs[0]
#test_embedding = model.encode([test_doc])
#print(f"Sample embedding created: {len(test_embedding[0])} dimensions")

# Add test document to collection
collection.add(
    documents=docs,
    embeddings=embeddings.tolist(),
    ids=doc_ids
)

print(f"Stored {len(docs)} documents")

# Verify storage
count = collection.count()
print(f"Vector database contains {count} documents")

# Show sample document
sample_doc = docs[0][:100] + "..." if len(docs[0]) > 100 else None


# Create completion marker
with open("documents_stored.txt", "w") as f:
    f.write(f"Stored {count} documents in vector database")



# Test Vector Search
query = "what is the cancellation policy?"

results = collection.query(
    query_texts=[query],
    n_results=2
)

print(f"Query: '{query}'")
for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
    similarity = 1 - distance
    print(f" {i*1}. Similarity: {similarity:.3f} - {doc}")