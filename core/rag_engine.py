import chromadb
from chromadb.utils import embedding_functions
from duckduckgo_search import DDGS

import hashlib
from datetime import datetime
from typing import List, Dict, Any

try:
    from sentence_transformers import CrossEncoder
    HAS_RERANKER = True
except Exception:
    HAS_RERANKER = False

def _stable_id_from_text(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()



class RAGengine:
    def __init__(self, persist_dir: str = "data/vector_store", use_reranker: bool = True):
        # Persistent Chroma client
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Better free embedding model
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="pico_rag",
            embedding_function=self.embedding_fn
        )

        # Optional reranker (free) for better precision
        self.use_reranker = use_reranker and HAS_RERANKER
        self.reranker = None
        if self.use_reranker:
            # Lightweight and fast; great quality boost
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ---------- Helpers ----------

    def _upsert_docs(self, docs: List[str], source: str) -> None:
        """
        Upsert docs into Chroma with stable SHA1 IDs and metadata.
        """
        if not docs:
            return

        ids = []
        metadatas = []
        for d in docs:
            doc_id = _stable_id_from_text(d)
            ids.append(doc_id)
            metadatas.append({
                "source": source,
                "added_at": datetime.utcnow().isoformat() + "Z"
            })

        # Chroma supports upsert in recent versions; if not, fallback to add with try/except
        try:
            self.collection.upsert(documents=docs, ids=ids, metadatas=metadatas)
        except AttributeError:
            # Older Chroma: emulate upsert by trying add, ignore duplicates
            try:
                self.collection.add(documents=docs, ids=ids, metadatas=metadatas)
            except Exception:
                # If some IDs already exist, add those that don't
                for d, i, m in zip(docs, ids, metadatas):
                    try:
                        self.collection.add(documents=[d], ids=[i], metadatas=[m])
                    except Exception:
                        pass  # duplicate; skip

    def _dedup_preserve_order(self, items: List[str]) -> List[str]:
        seen = set()
        out = []
        for it in items:
            key = _stable_id_from_text(it)
            if key not in seen:
                seen.add(key)
                out.append(it)
        return out

    def _maybe_rerank(self, query: str, docs: List[str], keep_top_k: int = 3) -> List[str]:
        """
        Apply reranking only if model available and there are enough docs to benefit.
        """
        if not self.use_reranker or not docs or len(docs) <= keep_top_k:
            return docs[:keep_top_k]

        pairs = [[query, d] for d in docs]
        scores = self.reranker.predict(pairs)  # higher is better
        ranked = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
        return ranked[:keep_top_k]

    # ---------- Retrieval steps ----------

    def search_local(self, query: str, top_k: int = 8) -> List[str]:
        """
        Semantic search in local vector store.
        """
        print("🔎 search_local")
        res = self.collection.query(query_texts=[query], n_results=top_k)
        if not res or not res.get("documents"):
            return []
        return res["documents"][0] or []

    def search_duckduckgo(self, query: str, num_results: int = 6) -> List[str]:
        """
        Free web fallback. We store short, useful snippets with URLs.
        """
        print("🌐 search_duckduckgo")
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=num_results)]
        # Normalize into readable lines
        docs = []
        for r in results:
            title = r.get("title") or ""
            body = r.get("body") or ""
            href = r.get("href") or ""
            snippet = f"{title} — {body} ({href})".strip()
            if snippet:
                docs.append(snippet)
        return docs

    # ---------- Public APIs ----------

    def add_to_db(self, docs: List[str], source_label: str = "manual") -> None:
        self._upsert_docs(docs, source=source_label)

    def add_document(self, doc_id: str, text: str, source_label: str = "manual") -> None:
        # Respect caller’s custom ID, but still store metadata
        try:
            self.collection.upsert(documents=[text], ids=[doc_id], metadatas=[{
                "source": source_label,
                "added_at": datetime.utcnow().isoformat() + "Z"
            }])
        except AttributeError:
            try:
                self.collection.add(documents=[text], ids=[doc_id], metadatas=[{
                    "source": source_label,
                    "added_at": datetime.utcnow().isoformat() + "Z"
                }])
            except Exception:
                # Fallback to update if add fails due to duplicate
                try:
                    self.collection.update(documents=[text], ids=[doc_id], metadatas=[{
                        "source": source_label,
                        "added_at": datetime.utcnow().isoformat() + "Z"
                    }])
                except Exception:
                    pass

    def retrieve(self, query: str, final_k: int = 3) -> List[str]:
        """
        1) Pull from local RAG
        2) If not enough, fetch from web and merge
        3) Deduplicate
        4) (Optional) Rerank
        5) Return top-k docs to the caller (ConversationEngine)
        """
        print("🚚 retrieve")

        local_docs = self.search_local(query, top_k=max(8, final_k))
        merged = list(local_docs)

        # If thin context, enrich from the web
        if len(merged) < final_k:
            web_docs = self.search_duckduckgo(query, num_results=6)
            # Store for future queries
            if web_docs:
                self._upsert_docs(web_docs, source="duckduckgo")
            merged.extend(web_docs)

        # Dedup & trim
        merged = self._dedup_preserve_order(merged)

        # Rerank for precision (if applicable)
        merged = self._maybe_rerank(query, merged, keep_top_k=final_k)

        return merged