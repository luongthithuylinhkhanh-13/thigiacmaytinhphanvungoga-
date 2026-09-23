import os
import glob
import numpy as np

class RAGEngine:
    def __init__(self, knowledge_dir="knowledge", model_name="keepitreal/vietnamese-sbert"):
        self.knowledge_dir = knowledge_dir
        self.model_name = model_name
        self.chunks = []
        self.chunk_sources = []
        self.embeddings = None
        self.model = None
        self.is_loaded = False
        
        self.load_knowledge_base()
        self.init_embedder()

    def load_knowledge_base(self):
        """Đọc và cắt nhỏ tài liệu từ thư mục knowledge/"""
        self.chunks = []
        self.chunk_sources = []
        
        if not os.path.exists(self.knowledge_dir):
            print(f"[!] Warning: Knowledge directory '{self.knowledge_dir}' does not exist.")
            return

        txt_files = glob.glob(os.path.join(self.knowledge_dir, "*.txt"))
        if not txt_files:
            print(f"[!] Warning: No .txt files found in '{self.knowledge_dir}'.")
            return

        for filepath in txt_files:
            filename = os.path.basename(filepath)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                for para in paragraphs:
                    self.chunks.append(para)
                    self.chunk_sources.append(filename)
            except Exception as e:
                print(f"[!] Error reading knowledge file {filename}: {e}")

        print(f"[+] Loaded {len(self.chunks)} knowledge chunks from {len(txt_files)} files in '{self.knowledge_dir}'.")

    def init_embedder(self):
        """Khởi tạo mô hình Embedding SentenceTransformer (Việt hóa)"""
        if not self.chunks:
            return

        try:
            from sentence_transformers import SentenceTransformer
            print(f"[*] Loading SentenceTransformer embedding model '{self.model_name}'...")
            self.model = SentenceTransformer(self.model_name)
            self.embeddings = self.model.encode(self.chunks, show_progress_bar=False, convert_to_numpy=True)
            self.is_loaded = True
            print("[+] SentenceTransformer embeddings indexed successfully.")
        except Exception as e:
            print(f"[!] Warning: Could not load SentenceTransformer '{self.model_name}': {e}")
            print("[*] Falling back to lightweight TF-IDF / Keyword match embedder...")
            self.init_fallback_embedder()

    def init_fallback_embedder(self):
        """Fallback embedder bằng TF-IDF nếu không thể tải model HuggingFace / offline"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model = TfidfVectorizer().fit(self.chunks)
            self.embeddings = self.model.transform(self.chunks).toarray()
            self.is_loaded = True
            print("[+] Fallback TF-IDF embedder indexed successfully.")
        except Exception as e:
            print(f"[!] Fallback embedder error: {e}")
            self.is_loaded = False

    def search(self, query, top_k=3, min_score=0.15):
        """
        Tìm kiếm ngữ cảnh liên quan nhất bằng Cosine Similarity (NumPy)
        Trả về: list các dict {"text": chunk, "source": file, "score": similarity}
        """
        if not self.chunks or not self.is_loaded or self.embeddings is None:
            return []

        query = query.strip()
        if not query:
            return []

        try:
            if hasattr(self.model, "encode"):
                query_vec = self.model.encode([query], convert_to_numpy=True)[0]
                norm_query = np.linalg.norm(query_vec)
                norm_embeds = np.linalg.norm(self.embeddings, axis=1)
                
                denom = norm_embeds * norm_query
                denom[denom == 0] = 1e-8
                
                similarities = np.dot(self.embeddings, query_vec) / denom
            else:
                query_vec = self.model.transform([query]).toarray()[0]
                norm_query = np.linalg.norm(query_vec)
                norm_embeds = np.linalg.norm(self.embeddings, axis=1)
                denom = norm_embeds * norm_query
                denom[denom == 0] = 1e-8
                similarities = np.dot(self.embeddings, query_vec) / denom

            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score >= min_score:
                    results.append({
                        "text": self.chunks[idx],
                        "source": self.chunk_sources[idx],
                        "score": score
                    })
            return results

        except Exception as e:
            print(f"[!] Error during semantic search: {e}")
            return []
