import sys
import requests
import json
import socket

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def run_diagnostics():
    print("==================================================")
    print("🔍 DIAGNOSTIC STEP 1 & 2: LM STUDIO / LOCAL LLM TEST")
    print("==================================================")

    urls_to_test = [
        "http://localhost:1234/v1/models",
        "http://127.0.0.1:1234/v1/models",
        "http://localhost:11434/v1/models",
        "http://127.0.0.1:11434/v1/models",
    ]

    working_endpoint = None
    real_model_id = None

    for url in urls_to_test:
        print(f"\n[*] Testing GET {url} ...")
        try:
            r = requests.get(url, timeout=3)
            print(f"    [+] HTTP Status: {r.status_code}")
            data = r.json()
            print(f"    [+] Response Data: {json.dumps(data, indent=2)}")
            
            models = data.get("data", [])
            if models and len(models) > 0:
                real_model_id = models[0].get("id")
                working_endpoint = url.rsplit('/models', 1)[0]
                print(f"    [SUCCESS] Found active model ID: '{real_model_id}' at '{working_endpoint}'")
                break
            else:
                print(f"    [!] Endpoint returned 200, but models list is EMPTY.")
        except requests.exceptions.ConnectionError as e:
            print(f"    [!] ConnectionError: Actively refused / not listening.")
        except Exception as e:
            print(f"    [!] Exception ({type(e).__name__}): {e}")

    print("\n==================================================")
    print("🔍 DIAGNOSTIC SUMMARY")
    print("==================================================")
    if working_endpoint and real_model_id:
        print(f"[LM STUDIO / LOCAL LLM]: SUCCESS")
        print(f"[WORKING ENDPOINT]: {working_endpoint}")
        print(f"[LM STUDIO MODEL ID]: {real_model_id}")
    else:
        print(f"[LM STUDIO / LOCAL LLM]: FAILED")
        print(f"[REASON]: No local LLM server is currently listening on port 1234 or port 11434 with a loaded model.")

if __name__ == "__main__":
    run_diagnostics()
