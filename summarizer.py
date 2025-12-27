from llm_helper import query_llm

def summary(text: str, config: dict) -> str:
    prompt = f"""Summarize this academic abstract in 3 bullet points.
Preserve key methods, datasets, and results.
Text:
{text}
"""
    return query_llm(prompt, config).strip()

def batch_summary(abstracts: list[str], config: dict) -> list[str]:
    # In a real production app, we would parallelize this.
    # For now, sequential is fine since we do "On Demand" summary in the new app.py
    return [summary(t, config) for t in abstracts]

def summarize_cluster(texts: list[str], style: str, config: dict) -> str:
    joined_text = "\n\n".join(texts[:10]) # Limit to top 10 to avoid token overflow
    
    if style == "Bullets":
        prompt_style = "a concise bulleted list"
    else:
        prompt_style = "a cohesive 2-paragraph summary"

    prompt = f"""You are a senior researcher. Synthesize the following abstracts into {prompt_style}. 
    Focus on common themes, methodologies, and findings.
    
    Abstracts:
    {joined_text}
    """
    
    return query_llm(prompt, config)