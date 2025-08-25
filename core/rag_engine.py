import chromadb
from chromadb.utils import embedding_functions
from duckduckgo_search import DDGS


class RAGengine:
    def __init__(self, persist_dir="data/vector_store"):
        # Persistent Chroma client
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Use sentence-transformers for embeddings (small + fast)
        # If you want Ollama embeddings, we’ll need to write a wrapper
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            # model_name="all-MiniLM-L6-v2"
             model_name="all-mpnet-base-v2"
            
        )

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="pico_rag",
            embedding_function=self.embedding_fn
        )

    def search_local(self, query: str, top_k=3):
        print("search local")
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return results["documents"][0] if results and results["documents"] else []

    # Search engine (DuckDuckGo)
    def search_duckduckgo(self, query: str, num_results=3):
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=num_results)]
        docs = [f"{r['title']} - {r['body']} ({r['href']})" for r in results]
        return docs

    def add_to_db(self, docs, query):
        for i, doc in enumerate(docs):
            self.collection.add(
                documents=[doc],
                ids=[f"{query}_{i}"]
            )

    def add_document(self, doc_id: str, text: str):
        self.collection.add(
            documents=[text],
            ids=[doc_id],
        )

    def retrieve(self, query: str):
        # First check local DB
        print("retrive function")
        docs = self.search_local(query)

        if docs:
            print("✅ Found in local DB")
            return docs

        print("🌐 Fetching from DuckDuckGo...")
        docs = self.search_duckduckgo(query)

        if docs:
            self.add_to_db(docs, query)

        return docs
