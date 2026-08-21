import os
import logging
from langchain_core.documents import Document

# Attempt to import necessary modules, handling missing ones gracefully
try:
    from langchain_community.vectorstores import Qdrant
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_groq import ChatGroq
    from langchain_cohere import CohereRerank
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
except ImportError:
    pass

def initialize_knowledge_base():
    """Initializes the Advanced RAG layer with Hybrid Search, Reranking, and Query Decomposition."""
    
    # 1. Document Ingestion (Mocking ingestion of Travel Guides, PDFs, Blogs)
    # In a real scenario, you would use DocumentLoaders (e.g., PyPDFLoader, WebBaseLoader)
    # and TextSplitters (e.g., RecursiveCharacterTextSplitter) to chunk documents.
    documents = [
        Document(page_content="Bali is a fantastic destination for budget travelers. You can enjoy local warung food for cheap.", metadata={"source": "bali_budget_guide"}),
        Document(page_content="When in Paris, booking Eiffel Tower tickets in advance is highly recommended to avoid long queues.", metadata={"source": "paris_tips"}),
        Document(page_content="Japan rail pass is essential for traveling across multiple cities in Japan cost-effectively.", metadata={"source": "japan_guide"}),
        Document(page_content="Best time to visit Thailand is between November and early April when the weather is cool and dry.", metadata={"source": "thailand_blog"}),
        Document(page_content="For a cheap trip to Europe, consider Eastern Europe like Budapest and Prague where costs are significantly lower.", metadata={"source": "europe_budget_pdf"}),
    ]
    
    # Setup Embeddings (text-embedding-3-small or open-source BGE)
    if os.environ.get("OPENAI_API_KEY"):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        # Fallback to Open Source BGE
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    
    # 2. Vector DB: Qdrant
    # In a real app, you would connect to a remote Qdrant, Pinecone, or Weaviate instance.
    qdrant_vectorstore = Qdrant.from_documents(
        documents,
        embeddings,
        location=":memory:",  # Using in-memory for demonstration
        collection_name="travel_knowledge"
    )
    vector_retriever = qdrant_vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 3. Hybrid Search (Semantic + BM25 Keyword Search)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5
    
    # Combine both retrievers using EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
    )
    
    # 4. Reranking (Cohere Rerank or cross-encoder)
    if os.environ.get("COHERE_API_KEY"):
        compressor = CohereRerank(top_n=3)
        reranker_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )
        base_search_retriever = reranker_retriever
    else:
        # Fallback if no Cohere API key
        base_search_retriever = ensemble_retriever
        
    # 5. Query Rewriting / Decomposition
    # Breaking down vague user queries ("plan a cheap trip") into sub-queries.
    if os.environ.get("OPENAI_API_KEY"):
        llm_rewriter = ChatOpenAI(temperature=0)
    elif os.environ.get("GROQ_API_KEY"):
        llm_rewriter = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    else:
        return base_search_retriever # Fallback if no LLM for rewriting
        
    advanced_retriever = MultiQueryRetriever.from_llm(
        retriever=base_search_retriever, 
        llm=llm_rewriter
    )
    
    return advanced_retriever

# Cache the retriever so it doesn't re-initialize on every tool call
_RETRIEVER = None

def get_travel_knowledge(query: str) -> str:
    """Useful to search curated travel knowledge base (PDFs, blogs, travel guides)."""
    global _RETRIEVER
    try:
        if _RETRIEVER is None:
            _RETRIEVER = initialize_knowledge_base()
            
        docs = _RETRIEVER.invoke(query)
        if not docs:
            return "No relevant information found in the curated travel knowledge base."
            
        return "\n\n".join([f"Source ({doc.metadata.get('source', 'Unknown')}): {doc.page_content}" for doc in docs])
    except Exception as e:
        logging.error(f"Error in RAG layer: {e}")
        return f"Could not retrieve from knowledge base due to an error: {e}. Please use general knowledge."
