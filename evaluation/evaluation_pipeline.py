from credentials import OPENAI_API_KEY
from openai import OpenAI
from pathlib import Path
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import json
from notebooks.pipeline import doc_loading_and_chunking, vector_database_setup, query_processing, vector_search
from config import CHUNKING_METHODS


def evaluation_pipeline():
    """
    This function evaluates the retrieval performance of different chunking methods on a set of questions.
    It loads the documents, creates chunks, sets up a vector database, processes queries, performs vector similarity search,
    and records the results for each chunking method.
    """
    with open("./evaluation/eval_dataset.json", "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data["questions"])

    chunk_method = CHUNKING_METHODS.keys()
    N = len(df)

    records = []

    for method in chunk_method:
        
        # Step 1: Loading documents & creating chunks
        chunks, chunk_sources = doc_loading_and_chunking("corpus", CHUNKING_METHODS[method])
        # Step 2: Setup the Vector Database
        collection = vector_database_setup(chunks, chunk_sources, method)
        
        for i in range(0, N):

            query = df['question'][i]
            print(f"\nQuestion: {query}")
            
            # Step 3: Process the Query
            model, query_embedding = query_processing(query)
            # Step 4: Vector Similarity Search
            search_results, search_sources = vector_search(collection, query_embedding, top_k=2)
            
            print(f"\nChunk Predicted: {search_results}")
            
            for rank, (chunk_text, source_meta) in enumerate(zip(search_results, search_sources), 1):
                records.append({
                    "question_id": df['id'][i],
                    "chunking_method": method,
                    "rank": rank,
                    "chunk_text": chunk_text,
                    "retrieved_source": source_meta["source"],
                    "true_source_docs": df['source_documents'][i],
                })
                
                
    results_df = pd.DataFrame(records)
    results_df.to_parquet("./evaluation/retrieval_results.parquet")
            
        
    return results_df



if __name__ == "__main__":
    results_df = evaluation_pipeline()
    print("\nEvaluation completed. Results saved to 'retrieval_results.parquet'.")
    



