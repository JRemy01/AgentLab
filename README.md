# AgentLab

A small benchmark for **grounded question answering**: an agent answers questions using only a set of source documents, and an LLM judge scores each answer for correctness, faithfulness, citations, completeness, and whether the agent declines when the documents don't contain the answer.

It runs the same benchmark against a hosted model on [Groq](https://groq.com) and a local model on [Ollama](https://ollama.com), so you can compare them side by side.

## How it works

```
data/benchmark.json ──► SingleAgent (Groq / Ollama) ──► answer
                                                          │
                                LLMJudge (Groq) ◄─────────┘
                                      │
                                      ▼
                        results/benchmark_results.json
```

1. **Benchmark cases** ([data/benchmark.json](data/benchmark.json)) each contain a question, source documents with IDs (`D1`, `D2`, …), the expected facts, and an `answerable` flag.
2. **The agent** ([agents/single_agent.py](agents/single_agent.py)) is told to answer only from the documents, cite document IDs like `[D1]`, and reply `"Insufficient information in the provided documents."` when the answer isn't there.
3. **The judge** ([evaluation/llm_judge.py](evaluation/llm_judge.py)) scores each answer from 0 to 1 and returns strict JSON, which is validated before it is accepted:

   | Metric              | Meaning                                                         |
   | ------------------- | --------------------------------------------------------------- |
   | `correctness`       | Factually accurate and relevant                                 |
   | `faithfulness`      | Claims are supported by the documents                           |
   | `citation_validity` | Citations exist and support the claims                          |
   | `completeness`      | Important supported facts are included                          |
   | `admit_wrong`       | For unanswerable questions: 1 if the agent declined, 0 if it made something up. `null` for answerable questions |
   | `reason`            | Short explanation of the scores                                 |

## Project structure

```
agentlab/
├── agents/
│   └── single_agent.py      # Grounded QA agent with citation rules
├── data/
│   └── benchmark.json       # Benchmark cases
├── evaluation/
│   └── llm_judge.py         # LLM-as-judge scoring and output validation
├── models/
│   ├── model.py             # LLMService protocol and LLMResponse dataclass
│   ├── groq_model.py        # GroqService (hosted)
│   └── ollama_model.py      # OllamaServise (local)
├── results/                 # Benchmark output
├── tests/
│   ├── run_benchmark.py     # Runs the full benchmark
│   └── test_services.py     # Quick check of each service
├── test_connection.py       # Raw connectivity check for Groq and Ollama
├── test_single_agent.py     # Runs the agent on one sample question
└── requirements.txt
```

## Setup

**Requirements:** Python 3.10+, a [Groq API key](https://console.groq.com/keys), and [Ollama](https://ollama.com) running locally.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download a local model for Ollama
ollama pull <model-name>
```

Create a `.env` file in the project root (it is gitignored):

```env
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=openai/gpt-oss-20b

OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=<model-name>
```

## Usage

Run all commands from the project root. Scripts inside `tests/` must be run as modules (`python -m ...`) so their imports resolve.

```bash
# 1. Check that both providers respond
python test_connection.py

# 2. Check the service wrappers
python -m tests.test_services

# 3. Try the agent on a single question
python test_single_agent.py

# 4. Run the full benchmark
python -m tests.run_benchmark
```

The benchmark writes `results/benchmark_results.json`. Each record contains the case ID, service, model, answer, latency, token counts, the judge's scores, and any error. A failure in one case is recorded and the run continues.

## Adding benchmark cases

Add entries to [data/benchmark.json](data/benchmark.json):

```json
{
  "id": "topic_004",
  "question": "Your question here?",
  "documents": [
    { "id": "D1", "text": "Source text the answer must come from." }
  ],
  "expected_facts": ["fact the answer should contain"],
  "answerable": true
}
```

To test whether the agent declines correctly, set `"answerable": false` and leave `expected_facts` empty.

## Adding a model provider

Any class with a `generate(prompt, system_prompt, temperature, max_tokens)` method that returns an `LLMResponse` satisfies the `LLMService` protocol in [models/model.py](models/model.py). Add it to the `services` list in [tests/run_benchmark.py](tests/run_benchmark.py) to include it in the benchmark.
