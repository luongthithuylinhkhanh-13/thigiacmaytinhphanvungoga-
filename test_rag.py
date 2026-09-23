import sys
import os

# Tự động reconfigure stdout sang UTF-8 để hiển thị tiếng Việt trên Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def test_rag_pipeline():
    print("--------------------------------------------------")
    print("[TEST] TESTING RAG PIPELINE & KNOWLEDGE BASE SEARCH")
    print("--------------------------------------------------")

    from rag.rag_engine import RAGEngine

    engine = RAGEngine(knowledge_dir="knowledge")

    if not engine.chunks:
        print("[!] FAILED: Knowledge base is empty!")
        sys.exit(1)

    print(f"[+] Loaded {len(engine.chunks)} chunks from knowledge base.")

    # Test Query 1: Khái niệm ổ gà
    query1 = "Ổ gà là gì và nguyên nhân hình thành?"
    results1 = engine.search(query1, top_k=3)
    print(f"\n[SEARCH] Query 1: '{query1}'")
    for r in results1:
        print(f"  - [{r['source']}] Score: {r['score']:.4f}\n    Content snippet: {r['text'][:80]}...")

    assert len(results1) > 0, "Semantic search should return results for Query 1"

    # Test Query 2: Mức độ nguy hiểm
    query2 = "Khi nào ổ gà ở mức độ nguy hiểm cao?"
    results2 = engine.search(query2, top_k=3)
    print(f"\n[SEARCH] Query 2: '{query2}'")
    for r in results2:
        print(f"  - [{r['source']}] Score: {r['score']:.4f}\n    Content snippet: {r['text'][:80]}...")

    assert len(results2) > 0, "Semantic search should return results for Query 2"

    print("\n[SUCCESS] RAG ENGINE TEST PASSED SUCCESSFULLY!\n")

if __name__ == "__main__":
    test_rag_pipeline()
