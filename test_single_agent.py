from agents.single_agent import SingleAgent
from models.groq_model import GroqService
from models.ollama_model import OllamaServise


DOCUMENTS = """
[D1] Distillation trains a student model to imitate a
teacher model. It can reduce inference cost and latency,
but the student may have lower performance.
"""

QUESTION = "Give one benefit and one limitation of distillation."

def main():
    services = [GroqService(), OllamaServise()]

    for service in services:
        agent = SingleAgent(service)

        print(f"\n=== {service.__class__.__name__} ===")

        try:
            result = agent.run(QUESTION,DOCUMENTS)

            print(result.text)
            print(f"Latency: {result.latency_seconds:.2f}s")
            print(f"Input tokens: {result.input_tokens}")
            print(f"Output tokens: {result.output_tokens}")

        except Exception as exc:
            print(f"Error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
