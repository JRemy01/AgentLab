import os,json
import requests
from dotenv import load_dotenv
from groq import Groq
from time import perf_counter

load_dotenv()

prompt = (
    "Answer in one short sentence: "
    "What is the main purpose of an AI benchmark?"
)


def test_groq():
    groq_api = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL")

    try:
        client = Groq(api_key=groq_api)
        start = perf_counter()

        response = client.chat.completions.create(
            model = groq_model,
            messages = [{"role":"user", "content":prompt}],
            max_tokens= 80,
            temperature=0,
        )

        latency = perf_counter() - start
        answer = response.choices[0].message.content

        print(f"\n[OK] Groq | model={groq_model}")
        print(f"Latency: {latency:.2f}s")
        print(f"Answer: {answer}")

    except Exception as e:
        print(f"\n[ERROR] Groq: {type(e).__name__}: {e}")



def test_ollama():
    ollama_url = os.getenv("OLLAMA_URL")
    ollama_model = os.getenv("OLLAMA_MODEL")
    start = perf_counter()
    try:
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model" : ollama_model,
                "messages": [{"role":"user", "content":prompt}],
                "stream": False,
                "options": {"temperature":0, "num_predict":80},
            },
            timeout= 120,
        )
        response.raise_for_status()

        latency = perf_counter() - start
        result = response.json()
        answer = result["message"]["content"]

        print(f"\n[OK] Ollama | model={ollama_model}")
        print(f"Latency: {latency:.2f}s")
        print(f"Answer: {answer}")

    except Exception as e:
        print(f"\n[ERROR] Ollama: {type(e).__name__}: {e}")





if __name__ == "__main__":
    test_groq()
    test_ollama()