from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np

corpus_dir = Path("corpus")
doc_paths = [str(p) for p in sorted(corpus_dir.rglob("*.md"))][1:]
docs = [Path(p).read_text(encoding="utf-8") for p in doc_paths]


queries = [
    "what is the cancellation policy?", 
    "Which is the subscription plan for the gym?",
    "What are the main facilities of the gym?"
    ]

model = SentenceTransformer('all-MiniLM-L6-v2')

doc_emeddings = model.encode(docs)

for query in queries:
    print(f"Searching for: {query}")
    
    query_embedding = model.encode([query])
    
    similarities = np.dot(query_embedding, doc_emeddings.T)
    top_indices = similarities.argsort()[0][-3:][::-1]
    
    print("Results:")
    
    for i, idx in enumerate(top_indices,1):
        doc_name = doc_paths[idx].split("/")[-1]
        print(f". {i}. Score: {similarities[0][idx]:.4f} - {doc_name}")
        
    print()
    
    