import requests

OLLAMA_URL = "http://localhost:11434/api/generate"   # Ollama default

def ask_llm(prompt: str) -> str:
    try:
        payload = {
            "model": "llama3.2:3b",   # your downloaded model
            "prompt": prompt,
            "stream": False          # IMPORTANT: no streaming → valid JSON
        }

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        return data.get("response", "<<No response>>")

    except Exception as e:
        return f"Error communicating with LLM: {e}"
