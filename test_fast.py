import sys
import os
import requests
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def run_fast_test():
    print("==================================================")
    print("🧪 FAST TEST DIAGNOSTICS: A -> F")
    print("==================================================")

    # 1. Test GET /v1/models
    print("\n--- TEST A: GET /v1/models ---")
    active_base = None
    active_model = None

    for url in ["http://localhost:1234/v1/models", "http://127.0.0.1:1234/v1/models", "http://localhost:11434/v1/models"]:
        try:
            r = requests.get(url, timeout=1.5)
            print(f"  GET {url} -> Status {r.status_code}")
            data = r.json()
            models = data.get("data", []) or data.get("models", [])
            if models and len(models) > 0:
                active_model = models[0].get("id")
                active_base = url.rsplit('/models', 1)[0]
                print(f"  [+] SUCCESS: Found Model ID '{active_model}' at '{active_base}'")
                break
            else:
                print(f"  [!] Connected to {url}, but model list is empty.")
        except Exception as e:
            print(f"  [!] Connection failed for {url}: {type(e).__name__}")

    # 2. Test RAG Search
    print("\n--- TEST C: RAG Semantic Search ---")
    from rag.rag_engine import RAGEngine
    engine = RAGEngine(knowledge_dir="knowledge")
    results = engine.search("Ổ gà là gì?", top_k=3)
    print(f"  Retrieved {len(results)} chunks:")
    for r in results:
        print(f"    - [{r['source']}] Score: {r['score']:.4f}: {r['text'][:50]}...")

    # 3. Test Chat Pipeline with YOLO Context
    print("\n--- TEST F: rag_chat with YOLO Context ---")
    from rag.chat import rag_chat
    mock_context = {
        "pothole_count": 4,
        "confidence": 88.5,
        "area": "85000",
        "risk_level": "Cao (Nghiêm trọng)"
    }
    ans = rag_chat("Ảnh trên có bao nhiêu ổ gà?", detection_context=mock_context)
    print(f"  Chatbot Response:\n  {ans}")

if __name__ == "__main__":
    run_fast_test()
