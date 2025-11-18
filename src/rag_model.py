import os
import time

try:
    from ollama import Ollama
    _HAS_OLLAMA_SDK = True
except Exception:
    _HAS_OLLAMA_SDK = False
    import subprocess

class GraphRAG:
    def __init__(self, model_name="deepseek-r1-8b"):
        self.model_name = model_name
        if _HAS_OLLAMA_SDK:
            self.client = Ollama(model=self.model_name)
        else:
            self.client = None

    def prompt_text(self, question, retrieved_passages, graph_summary, few_shot_examples=None):
        context = "\n\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(retrieved_passages)])
        prompt = f"""You are an evidence-based biomedical question-answering assistant.
Answer the question using only the provided evidence and graph summary.

Question:
{question}

Retrieved evidence:
{context}

Graph summary:
{graph_summary}

Task: Provide a single-word answer (yes / no / maybe) followed by a one-sentence justification that cites evidence numbers above.
Answer:"""

        return prompt

    def call(self, prompt):
        if _HAS_OLLAMA_SDK:
            resp = self.client.prompt(prompt)
            return resp
        else:
            proc = subprocess.run(["ollama", "run", self.model_name, prompt], capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"Ollama CLI error: {proc.stderr}")
            return proc.stdout.strip()

    def answer(self, question, retrieved_passages, graph_summary, few_shot_examples=None):
        prompt = self.prompt_text(question, retrieved_passages, graph_summary, few_shot_examples)
        return self.call(prompt)
