import sys
import os

# Tự động reconfigure stdout sang UTF-8 để hiển thị tiếng Việt trên Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def test_chat_pipeline():
    print("--------------------------------------------------")
    print("[TEST] TESTING RAG CHAT FUNCTION & ERROR HANDLING")
    print("--------------------------------------------------")

    from rag.chat import rag_chat

    question = "Ảnh này có bao nhiêu ổ gà và diện tích thế nào?"
    yolo_context = "Số ổ gà: 3\nTổng diện tích: 85000 pixels\nConfidence trung bình: 88.5%\nĐánh giá Risk: Nguy cơ trung bình"

    print(f"[CHAT] Sending Question: '{question}'")
    response = rag_chat(question, detection_context=yolo_context)

    print("\n[CHATBOT RESPONSE]:")
    print(response)

    assert isinstance(response, str) and len(response) > 0, "Chatbot response must be a non-empty string"
    print("\n[SUCCESS] CHAT PIPELINE TEST COMPLETED SUCCESSFULLY!\n")

if __name__ == "__main__":
    test_chat_pipeline()
