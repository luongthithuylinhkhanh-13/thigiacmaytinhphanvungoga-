import os
import requests
import traceback

# Mặc định kết nối LM Studio local
PRIMARY_LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
FALLBACK_OLLAMA_URL = "http://localhost:11434/v1"
LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "")

def check_local_llm_status():
    """
    Kiểm tra thực tế các endpoint local LLM (/v1/models).
    Trả về: (status_code, base_url, model_id_hoac_thong_bao)
    - status_code: "OK", "NO_MODEL", "SERVER_DOWN"
    """
    endpoints = [
        PRIMARY_LM_STUDIO_URL,
        "http://127.0.0.1:1234/v1",
        FALLBACK_OLLAMA_URL,
        "http://127.0.0.1:11434/v1"
    ]

    for base_url in endpoints:
        models_url = f"{base_url.rstrip('/')}/models"
        try:
            res = requests.get(models_url, timeout=2)
            if res.status_code == 200:
                data = res.json()
                models = data.get("data", []) or data.get("models", [])
                if models and len(models) > 0:
                    model_id = LM_STUDIO_MODEL if LM_STUDIO_MODEL else models[0].get("id")
                    if model_id:
                        print(f"[+] LOCAL LLM ONLINE! Endpoint: '{base_url}', Model: '{model_id}'")
                        return "OK", base_url, model_id
                else:
                    print(f"[!] Local LLM server at '{base_url}' is running (HTTP 200), but NO MODEL IS LOADED.")
                    return "NO_MODEL", base_url, "LM Studio / AI local đã chạy nhưng chưa có model nào được load."
        except requests.exceptions.ConnectionError:
            continue
        except Exception as e:
            print(f"[!] Error checking '{models_url}': {e}")
            continue

    print("================== CHATBOT CONNECTION ERROR ==================")
    print("Error Type: ConnectionError")
    print("Message: Cannot connect to LM Studio at http://localhost:1234/v1/models (Connection refused).")
    print("LM Studio Server is NOT STARTED. Please open LM Studio and click 'Start Server'.")
    print("==============================================================")
    
    return "SERVER_DOWN", PRIMARY_LM_STUDIO_URL, "Không thể kết nối đến AI local. Hãy mở LM Studio và Start Server."

def generate_llm_response(prompt_messages, temperature=0.2, max_tokens=512):
    """
    Gửi request POST /v1/chat/completions tới Local LLM Server
    """
    status, base_url, result = check_local_llm_status()

    if status == "SERVER_DOWN":
        return result
    if status == "NO_MODEL":
        return result

    model_name = result
    completions_url = f"{base_url.rstrip('/')}/chat/completions"

    payload = {
        "model": model_name,
        "messages": prompt_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }

    try:
        print(f"[*] Sending prompt to Local LLM ({completions_url}) with model '{model_name}'...")
        response = requests.post(completions_url, json=payload, timeout=20)

        if response.status_code == 404:
            print(f"[!] ERROR HTTP 404: Model '{model_name}' not found on {base_url}")
            return "Không tìm thấy model LM Studio được cấu hình."

        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        if choices and len(choices) > 0:
            message = choices[0].get("message", {})
            content = message.get("content", "").strip()
            if content:
                print("[+] Successfully received answer from Local LLM!")
                return content

        return "Không nhận được phản hồi hợp lệ từ mô hình AI local."

    except requests.exceptions.ConnectionError as e:
        print("================== CHATBOT ERROR ==================")
        print("Exception:", type(e).__name__)
        print("Details:", str(e))
        print("===================================================")
        return "Không thể kết nối đến AI local. Hãy mở LM Studio và Start Server."

    except requests.exceptions.Timeout as e:
        print("================== CHATBOT TIMEOUT ==================")
        print("Exception:", type(e).__name__)
        print("Details:", str(e))
        print("=====================================================")
        return "AI local phản hồi quá lâu. Hãy kiểm tra model trong LM Studio."

    except requests.exceptions.HTTPError as e:
        status_code = response.status_code if 'response' in locals() else 'Unknown'
        print("================== CHATBOT HTTP ERROR ==================")
        print("Status Code:", status_code)
        print("Details:", str(e))
        print("========================================================")
        if status_code == 404:
            return "Không tìm thấy model LM Studio được cấu hình."
        return f"Lỗi phản hồi từ AI local LM Studio (Mã lỗi: {status_code})."

    except Exception as e:
        print("================== CHATBOT UNEXPECTED ERROR ==================")
        print("Exception:", type(e).__name__)
        print("Details:", str(e))
        traceback.print_exc()
        print("==============================================================")
        return f"Lỗi hệ thống khi gọi AI local: {str(e)}"
