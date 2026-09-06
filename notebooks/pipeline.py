from credentials import OPENAI_API_KEY
from openai import OpenAI
from pathlib import Path
from langchain.text_splitter import MarkdownHeaderTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
from config import CHUNKING_METHODS

# ==========================
# 1. Document Loading and Chunking
# =========================

def doc_loading_and_chunking(folder_name, chunking_method):

    print(f"\n1. Loading the documents in {folder_name} and chunking them using {chunking_method}...")
    corpus_dir = Path(folder_name)
    doc_paths = [str(p) for p in sorted(corpus_dir.rglob("*.md"))][1:] # Exclude README file
    docs = [Path(p).read_text(encoding="utf-8") for p in doc_paths]
    doc_ids = [f"doc_{i+1}" for i in range(len(docs))]

    splitter = chunking_method
    
    chunks = []
    chunk_sources = []
    
    for doc, path in zip(docs, doc_paths):
        doc_chunk = splitter.split_text(doc)
        if isinstance(splitter, MarkdownHeaderTextSplitter):
            page_contents = [doc.page_content for doc in doc_chunk]
            chunks.extend(page_contents)
            chunk_sources.extend([path]*len(page_contents))
                        
        else:
            chunks.extend(doc_chunk)
            chunk_sources.extend([path]*len(doc_chunk))
        
    print(f"   > There is a total of {len(chunks)} chunks across {len(docs)} documents")

    return chunks, chunk_sources


# ==========================
# 2. Vector Database Setup
# =========================

def vector_database_setup(chunks, chunk_sources, method):

    print("\n2. Creating Vector Collection...")
    client = chromadb.Client()
    
    try:
        collection = client.create_collection(f"titanic-fitness-{method}")
    except:
        collection = client.get_collection(f"titanic-fitness-{method}")
        
    chunk_ids = [f"chunk_{i+1}" for i in range(len(chunks))]
    metadatas = [{"source": src} for src in chunk_sources]
    
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=chunk_ids
    )
    
    print("   > Vector Chunked Collection Stored")
    return collection


# ==========================
# 3. Query processing
# =========================

def query_processing(query):
    
    print("\n3. Processing Query...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    cleaned_query = query.lower().strip()
    query_embedding = model.encode([cleaned_query])
    
    print(f"   > Count Embedding dimensions {query_embedding.shape}")
    
    return model, query_embedding[0]


# ==========================
# 4. Vector Search
# =========================

def vector_search(collection, query_embedding, top_k):
    
    print("\n4. Vector Searching...")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    search_results = results['documents'][0]
    search_sources = results['metadatas'][0]
    similarity = 1- results['distances'][0][0]
    
    print(f"   > Found {top_k} result with similarity {similarity:.3f}")
    
    return search_results, search_sources

# ==========================
# 5. Context Augmentation
# =========================

def context_augmentation(query, search_results):
    
    print("\n5. Including Context to Prompt...")
    context = search_results[0]
    augmented_prompt = f"""
    Based on the following company policies, answer the user's question.
    
    POLICIES: {context}
    
    QUERY: {query}
    
    Please provide a clear, accurate answer based on the company policies.
    If the information is not available in the policies, say so.
    Include relevant policy details and any limitations or requirements.
    """

    return augmented_prompt


# ==========================
# 6. Response Generation
# =========================

def generate_response(augmented_prompt):
    
    print("\n6. Generating Response...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=augmented_prompt
    )

    return response.output_text


# ==========================
# 7. Complete RAG Pipeline
# =========================

def complete_rag_pipeline(query, method_name):

    chunking_method = CHUNKING_METHODS[method_name]

    # Step 1: Loading documents & creating chunks
    chunks, chunk_sources = doc_loading_and_chunking("corpus", chunking_method)
    # Step 2: Setup the Vector Database
    collection = vector_database_setup(chunks, chunk_sources, method_name)
    # Step 3: Process the Query
    model, query_embedding = query_processing(query)
    # Step 4: Vector Similarity Search
    search_results, search_sources = vector_search(collection, query_embedding, top_k=1)
    # Step 5: Prompt augmentation
    augmented_prompt = context_augmentation(query, search_results)
    # Step 6: Generate response
    response = generate_response(augmented_prompt)

    return response


if __name__ == "__main__":
    
    
    queries = [
        "what is the cancellation policy?", 
        "Which is the subscription plan for the gym?",
        "What are the main facilities of the gym?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n\n==================== Query {i} ====================")
        print(f"Query: {query}")
        response = complete_rag_pipeline(query, "markdown_header")
        print(f"\nResponse: {response}")
