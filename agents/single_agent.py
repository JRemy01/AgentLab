from models.model import LLMResponse

SYSTEM_PROMPT = """
You are a research assistant working with source documents.

Rules:
1. Answer using only the provided documents.
2. Never invent facts or citations.
3. If the documents do not contain the answer, say:
   "Insufficient information in the provided documents."
4. Cite supporting document IDs, such as [D1].
5. Separate supported facts from uncertainty.
"""


class SingleAgent:
    def __init__(self,service):
        self.service = service

    def run(self, question:str,documents:str)->LLMResponse:
        prompt= f"""
QUESTION:
{question}

SOURCE DOCUMENTS:
{documents}

Answer the question using the system rules.
"""
        return self.service.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=250,
        )