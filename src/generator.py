import requests
import config

def generate_summary(prompt: str, model_name: str = config.DEFAULT_LLM_MODEL, temperature: float = config.TEMPERATURE) -> str:
    """
    Mengirimkan prompt Context Stuffing ke LLM Ollama lokal untuk menghasilkan ringkasan multi-dokumen.
    """
    url = f"{config.OLLAMA_HOST}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 2048
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Gagal terhubung ke Ollama server di {config.OLLAMA_HOST}. Pastikan Ollama sudah berjalan (`ollama serve`).")
    except Exception as e:
        print(f"Error saat inferensi Ollama ({model_name}): {e}")
        return f"[ERROR GENERASI LLM: {e}]"
