from models.model import LLMService
from models.ollama_model import OllamaServise
from models.groq_model import GroqService

PROMPT = "Explain the purpose of an AI benchmark in 2 sentences."

def run_test(service):
    print(f"Testing {service.__class__.__name__}")

    result = service.generate(
        prompt=PROMPT,
        system_prompt="Answer concisely and accurately.",
        temperature=0.0,
        max_tokens=100,
    )

    print(f"Service: {result.service}")
    print(f"Model: {result.model}")
    print(f"Latency: {result.latency_seconds}")
    print(f"Input tokens: {result.input_tokens}")
    print(f"Output tokens: {result.output_tokens}")
    print(f"Answer: {result.text}")


if __name__ == "__main__":
    services = [GroqService(), OllamaServise()]

    for service in services:
        try:
            run_test(service)
        except Exception as e:
            print(f"ERROR {service.__class__.__name__}: "
                f"{type(e).__name__}: {e}"
            )
