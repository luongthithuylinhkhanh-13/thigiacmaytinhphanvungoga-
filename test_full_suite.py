import sys
import os
import requests
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def run_full_suite():
    print("==================================================")
    print("🧪 EXECUTING TEST SUITE: A -> F")
    print("==================================================")

    results_summary = {
        "LM_STUDIO": "FAIL",
        "MODEL_ID": "None",
        "RAG": "FAIL",
        "FLASK_CHAT": "FAIL",
        "YOLO_CONTEXT": "FAIL",
        "CHATBOT": "FAIL"
    }

    # ----------------------------------------------------
    # TEST A: GET /v1/models
    # ----------------------------------------------------
    print("\n--- TEST A: GET /v1/models ---")
    endpoints_to_try = ["http://localhost:1234/v1", "http://127.0.0.1:1234/v1", "http://localhost:11434/v1"]
    active_base = None
    active_model = None

    for ep in endpoints_to_try:
        url = f"{ep}/models"
        try:
            r = requests.get(url, timeout=2)
            print(f"  GET {url} -> Status: {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                models = data.get("data", []) or data.get("models", [])
                if models and len(models) > 0:
                    active_model = models[0].get("id")
                    active_base = ep
                    results_summary["LM_STUDIO"] = "PASS"
                    results_summary["MODEL_ID"] = active_model
                    print(f"  [+] TEST A PASSED! Active Model ID: '{active_model}' at '{active_base}'")
                    break
                else:
                    print(f"  [!] Endpoint {url} connected, but model list is EMPTY.")
        except Exception as e:
            print(f"  [!] Connection failed for {url}: {type(e).__name__}")

    # ----------------------------------------------------
    # TEST B: POST /v1/chat/completions
    # ----------------------------------------------------
    print("\n--- TEST B: POST /v1/chat/completions ---")
    if active_base and active_model:
        chat_url = f"{active_base}/chat/completions"
        payload = {
            "model": active_model,
            "messages": [{"role": "user", "content": "Xin chào, hãy trả lời bằng tiếng Việt ngắn gọn."}],
            "temperature": 0.2,
            "stream": False
        }
        try:
            r = requests.post(chat_url, json=payload, timeout=10)
            print(f"  STATUS CODE: {r.status_code}")
            print(f"  RESPONSE: {r.text[:200]}...")
            if r.status_code == 200:
                print("  [+] TEST B PASSED!")
            else:
                print("  [!] TEST B FAILED!")
        except Exception as e:
            print(f"  [!] TEST B EXCEPTION: {type(e).__name__} - {e}")
    else:
        print("  [!] SKIPPING TEST B: LM Studio server is not running or no model loaded.")

    # ----------------------------------------------------
    # TEST C: RAG Semantic Search
    # ----------------------------------------------------
    print("\n--- TEST C: RAG Semantic Search ('Ổ gà là gì?') ---")
    try:
        from rag.rag_engine import RAGEngine
        engine = RAGEngine(knowledge_dir="knowledge")
        chunks = engine.search("Ổ gà là gì?", top_k=3)
        print(f"  Retrieved {len(chunks)} chunks from knowledge/")
        for c in chunks:
            print(f"    - [{c['source']}] Score: {c['score']:.4f}: {c['text'][:60]}...")
        if len(chunks) > 0:
            results_summary["RAG"] = "PASS"
            print("  [+] TEST C PASSED!")
        else:
            print("  [!] TEST C FAILED: No chunks retrieved.")
    except Exception as e:
        print(f"  [!] TEST C EXCEPTION: {e}")

    # ----------------------------------------------------
    # TEST D, E, F: Chatbot pipeline tests via rag_chat
    # ----------------------------------------------------
    from rag.chat import rag_chat

    print("\n--- TEST D: rag_chat('Xin chào') ---")
    ans_d = rag_chat("Xin chào")
    print(f"  [CHATBOT D RESPONSE]: {ans_d}")

    print("\n--- TEST E: rag_chat('Ổ gà là gì?') ---")
    ans_e = rag_chat("Ổ gà là gì?")
    print(f"  [CHATBOT E RESPONSE]: {ans_e}")

    print("\n--- TEST F: rag_chat('Ảnh trên có bao nhiêu ổ gà?', detection_context) ---")
    yolo_mock = {
        "pothole_count": 4,
        "confidence": 88.5,
        "area": "85000",
        "risk_level": "Cao (Nghiêm trọng - Cần khắc phục gấp)"
    }
    ans_f = rag_chat("Ảnh trên có bao nhiêu ổ gà?", detection_context=yolo_mock)
    print(f"  [CHATBOT F RESPONSE]: {ans_f}")

    if "Không thể kết nối" not in ans_d and "Lỗi" not in ans_d:
        results_summary["FLASK_CHAT"] = "PASS"
        results_summary["CHATBOT"] = "PASS"

    if "4 ổ gà" in ans_f or "4" in ans_f:
        results_summary["YOLO_CONTEXT"] = "PASS"

    print("\n==================================================")
    print("📊 FINAL SUITE SUMMARY")
    print("==================================================")
    print(f"[LM STUDIO]     : {results_summary['LM_STUDIO']}")
    print(f"[MODEL]         : {results_summary['MODEL_ID']}")
    print(f"[RAG]           : {results_summary['RAG']}")
    print(f"[FLASK /chat]   : {results_summary['FLASK_CHAT']}")
    print(f"[YOLO CONTEXT]  : {results_summary['YOLO_CONTEXT']}")
    print(f"[CHATBOT]       : {results_summary['CHATBOT']}")
    print("==================================================")

if __name__ == "__main__":
    run_full_suite()
