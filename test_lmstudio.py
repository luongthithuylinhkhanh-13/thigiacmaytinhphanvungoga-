import sys
import requests
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def test_lm_studio_standalone():
    print("==================================================")
    print("[TEST] STANDALONE LM STUDIO CONNECTION TEST")
    print("==================================================")

    models_url = "http://localhost:1234/v1/models"
    chat_url = "http://localhost:1234/v1/chat/completions"

    # Step 1: Check models
    print(f"[*] Step 1: Querying GET {models_url} ...")
    try:
        res = requests.get(models_url, timeout=3)
        print(f"[+] Response Status Code: {res.status_code}")
        body = res.json()
        print(f"[+] Response Body: {json.dumps(body, indent=2)}")

        models = body.get("data", [])
        if not models or len(models) == 0:
            print("\n[!] DIAGNOSIS: LM Studio server is RUNNING on port 1234, but NO MODEL IS LOADED in LM Studio UI.")
            print("[!] SOLUTION: Open LM Studio UI, select a downloaded model at the top dropdown bar, and ensure it is loaded.")
            return False

        model_id = models[0].get("id")
        print(f"\n[+] Active Model ID detected: '{model_id}'")

    except requests.exceptions.ConnectionError as e:
        print(f"\n[!] DIAGNOSIS: Cannot connect to LM Studio on port 1234 ({e})")
        print("[!] REASON: LM Studio Server is NOT STARTED.")
        print("[!] SOLUTION: Open LM Studio -> Go to 'Developer / Local Server' tab -> Click 'Start Server'.")
        return False
    except Exception as e:
        print(f"\n[!] Unexpected Error: {e}")
        return False

    # Step 2: Test Chat Completion
    print(f"\n[*] Step 2: Testing POST {chat_url} with model '{model_id}' ...")
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Xin chào, hãy trả lời bằng tiếng Việt ngắn gọn."}
        ],
        "temperature": 0.2,
        "stream": False
    }

    try:
        res = requests.post(chat_url, json=payload, timeout=15)
        print(f"[+] Chat Completion Status Code: {res.status_code}")
        if res.status_code == 200:
            data = res.json()
            reply = data["choices"][0]["message"]["content"]
            print(f"\n[+] LLM RESPONSE:\n{reply}\n")
            print("[SUCCESS] LM STUDIO STANDALONE TEST PASSED 100%!")
            return True
        else:
            print(f"[!] Error Response: {res.text}")
            return False
    except Exception as e:
        print(f"[!] Chat completion error: {e}")
        return False

if __name__ == "__main__":
    test_lm_studio_standalone()
