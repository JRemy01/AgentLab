
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from agents.single_agent import SingleAgent
from evaluation.llm_judge import LLMJudge
from models.groq_model import GroqService
from models.ollama_model import OllamaServise


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "benchmark.json"
RESULTS_DIR = ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "benchmark_results.json"


def load_benchmark():
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    cases = load_benchmark()

    judge = LLMJudge(GroqService())

    services = [
        GroqService(),
        OllamaServise(),
    ]

    results = []

    for service in services:
        agent = SingleAgent(service)

        for case in cases:
            case_id = case["id"]
            documents = case["documents"]

            documents_text = "\n".join(
                f"[{doc['id']}] {doc['text']}"
                for doc in documents
            )

            print(f"Running {service.__class__.__name__}: {case_id}")

            record = {
                "case_id": case_id,
                "service": service.__class__.__name__,
                "model": service.model,
                "answer": None,
                "generation_latency_seconds": None,
                "input_tokens": None,
                "output_tokens": None,
                "evaluation": None,
                "error": None,
            }

            try:
                response = agent.run(
                    question=case["question"],
                    documents=documents_text,
                )

                record["answer"] = response.text
                record["generation_latency_seconds"] = (
                    response.latency_seconds
                )
                record["input_tokens"] = response.input_tokens
                record["output_tokens"] = response.output_tokens

                evaluation = judge.evaluate(
                    question=case["question"],
                    documents=documents,
                    answer=response.text,
                    answerable=case["answerable"],
                )

                record["evaluation"] = asdict(evaluation)

            except Exception as exc:
                record["error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                print(f"  ERROR: {record['error']}")

            results.append(record)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_file": str(DATA_PATH.relative_to(ROOT)),
        "case_count": len(cases),
        "result_count": len(results),
        "results": results,
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    successful = sum(
        record["error"] is None
        and record["evaluation"] is not None
        for record in results
    )

    print(f"\nCompleted: {successful}/{len(results)} evaluations")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
