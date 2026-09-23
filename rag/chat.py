import os
from .rag_engine import RAGEngine
from .llm import generate_llm_response, PRIMARY_LM_STUDIO_URL, LM_STUDIO_MODEL

# Khởi tạo RAG Engine toàn cục
rag_engine = RAGEngine(knowledge_dir="knowledge", model_name="keepitreal/vietnamese-sbert")

SYSTEM_PROMPT = """Bạn là Trợ lý AI thông minh tích hợp trong Hệ thống Phân vùng Ổ gà Mặt đường (Pothole Segmentation).

Nhiệm vụ của bạn:
1. Trả lời câu hỏi liên quan đến KẾT QUẢ PHÂN TÍCH ẢNH HIỆN TẠI (số lượng ổ gà, diện tích, confidence, risk score).
2. Giải thích các khái niệm, quy trình kỹ thuật, chỉ số đo lường (mAP50, IoU, NMS) và phương pháp sửa chữa ổ gà từ CƠ SỞ DỮ LIỆU KẾT NỐI (RAG Context).

Quy tắc bắt buộc:
- Trả lời hoàn toàn bằng tiếng Việt.
- Trả lời ngắn gọn, chính xác, dễ hiểu.
- KHÔNG TỰ BỊA ĐẶT THÔNG TIN.
- Nếu câu hỏi liên quan đến ảnh đang hiển thị (ví dụ: "ảnh này có bao nhiêu ổ gà", "diện tích ổ gà", "mức độ nguy hiểm"): BẮT BUỘC ưu tiên sử dụng KẾT QUẢ PHÂN TÍCH ẢNH HIỆN TẠI.
- Nếu người dùng hỏi về ảnh mà chưa có ảnh nào được phân tích, hãy thông báo rõ là chưa có kết quả phân tích ảnh.
- Nếu người dùng hỏi kiến thức chung mà trong CƠ SỞ DỮ LIỆU RAG không có thông tin phù hợp, trả lời chính xác:
"Hệ thống chưa có thông tin phù hợp trong cơ sở dữ liệu."
"""

def format_detection_context(detection_context):
    """
    Format dữ liệu kết quả YOLOv8-Seg thành chuỗi văn bản rõ ràng cho LLM
    """
    if not detection_context:
        return "Chưa có ảnh nào được phân tích hoặc chưa thực hiện phân vùng ổ gà."

    if isinstance(detection_context, dict):
        count = detection_context.get("pothole_count", 0)
        area = detection_context.get("area", "N/A")
        conf = detection_context.get("confidence", 0.0)
        risk = detection_context.get("risk_level", "Chưa xác định")

        if count == 0:
            return "KẾT QUẢ PHÂN TÍCH: Đã phân tích ảnh nhưng không phát hiện ổ gà nào."

        return (
            f"KẾT QUẢ PHÂN TÍCH ẢNH HIỆN TẠI:\n"
            f"- Tổng số ổ gà phát hiện: {count} ổ gà\n"
            f"- Độ tin cậy trung bình (Confidence): {conf:.1f}%\n"
            f"- Tổng diện tích phân vùng (Area): {area} pixels\n"
            f"- Đánh giá mức độ nguy hiểm (Risk Level): {risk}"
        )
    elif isinstance(detection_context, str):
        return f"KẾT QUẢ PHÂN TÍCH ẢNH HIỆN TẠI:\n{detection_context.strip()}"

    return str(detection_context)

def rag_chat(question, detection_context=None):
    """
    Hàm xử lý RAG Chat chính:
    1. Chuẩn hóa câu hỏi & dữ liệu YOLO detection
    2. Tìm kiếm top 3 đoạn kiến thức RAG từ knowledge/
    3. Xây dựng prompt đầy đủ gửi Local LLM
    """
    if not question or not str(question).strip():
        return "Xin lỗi, tôi chưa nhận được câu hỏi."

    question_text = str(question).strip()

    # 1. Semantic Search trong knowledge base
    rag_results = rag_engine.search(question_text, top_k=3, min_score=0.15)
    
    if rag_results:
        context_texts = [f"[{i+1}] ({r['source']}): {r['text']}" for i, r in enumerate(rag_results)]
        rag_context_str = "\n\n".join(context_texts)
    else:
        rag_context_str = "Không tìm thấy thông tin phù hợp trong cơ sở dữ liệu kiến thức RAG."

    # 2. Xử lý ngữ cảnh YOLO
    yolo_context_str = format_detection_context(detection_context)

    # In log debug ra Terminal theo yêu cầu
    print("\n================ [DEBUG CHAT REQUEST] ================")
    print("CHAT QUESTION:", question_text)
    print("DETECTION CONTEXT:", yolo_context_str)
    print("RAG CONTEXT:", rag_context_str[:200] + "..." if len(rag_context_str) > 200 else rag_context_str)
    print("LM STUDIO URL:", PRIMARY_LM_STUDIO_URL)
    print("LM STUDIO MODEL:", LM_STUDIO_MODEL if LM_STUDIO_MODEL else "(Auto-detect from server)")
    print("======================================================")

    # 3. Tạo Prompt hoàn chỉnh
    prompt_content = (
        f"--- KẾT QUẢ PHÂN TÍCH ẢNH HIỆN TẠI ---\n{yolo_context_str}\n\n"
        f"--- KIẾN THỨC RAG HỆ THỐNG ---\n{rag_context_str}\n\n"
        f"--- CÂU HỎI NGƯỜI DÙNG ---\n{question_text}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_content}
    ]

    # 4. Gửi yêu cầu đến LM Studio
    return generate_llm_response(messages)
