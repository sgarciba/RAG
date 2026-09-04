from credentials import OPENAI_API_KEY
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from pathlib import Path

corpus_dir = Path("corpus")
doc_paths = [str(p) for p in sorted(corpus_dir.rglob("*.md"))][1:]
docs = [Path(p).read_text(encoding="utf-8") for p in doc_paths]


queries = [
    "what is the cancellation policy?", 
    "Which is the susbcription plan for the gym?",
    "What are the main facilities of the gym?"
    ]


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)

for query in queries:
    print(f"Searching for: {query}")

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    top_indices = similarities.argsort()[0][-3:][::-1]
    
    print("Results:")
    
    for i, idx in enumerate(top_indices,1):
        doc_name = doc_paths[idx].split("/")[-1]
        print(f". {i}. Score: {similarities[0][idx]:.4f} - {doc_name}")
        
    print()



client = OpenAI(api_key=OPENAI_API_KEY)

response = client.responses.create(
    model="gpt-5.6-luna",
    input="Say hello! Explain in one sentence what an LLM is."
)

print(response.output_text)


