import os
import traceback
from rag.chat import rag_chat

def get_chat_response(message, context=""):
    """
    Hàm wrapper giữ tương thích ngược với Flask app.
    Ủy quyền hoàn toàn cho RAG Pipeline Local (rag_chat).
    """
    try:
        return rag_chat(message, detection_context=context)
    except Exception as e:
        print(f"❌ Error in get_chat_response: {e}")
        traceback.print_exc()
        return "Lỗi hệ thống khi xử lý câu hỏi. Vui lòng thử lại sau."
