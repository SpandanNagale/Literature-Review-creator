from groq import Groq
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

import ollama

def summary(text: str, model: str = "llama3.1:8b") -> str:
    """
    Summarize an academic abstract into 3 bullet points using a local Ollama model.
    """
    prompt = f"""Summarize this academic abstract in 3 bullet points.
Preserve key methods, datasets, and results.
Text:
{text}
"""

    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return resp["message"]["content"].strip()


def batch_summary(abstract:list[str])->list[str]:
    return [summary(t) for t in abstract]

def summarize_cluster(texts, style="Paragraph"):
    joined = " ".join(texts)
    if style == "Bullets":
        return "\n".join([f"- {t}" for t in texts[:5]])
    else:
        return f"In this theme, {joined[:]}..."


