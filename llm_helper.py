import os
import ollama
import google.generativeai as genai

def query_llm(prompt: str, config: dict) -> str:
    """
    Unified interface for querying LLMs (Ollama or Gemini).
    
    config dict structure:
    {
        "provider": "Ollama" or "Gemini",
        "model": "llama3" or "gemini-1.5-flash",
        "api_key": "..." (only for Gemini)
    }
    """
    provider = config.get("provider", "Ollama")
    model = config.get("model", "gemma3:latest")
    
    try:
        if provider == "Gemini":
            api_key = config.get("api_key")
            if not api_key:
                return "Error: Google API Key is missing."
            
            genai.configure(api_key=api_key)
            # gemini-1.5-flash is faster/cheaper for this use case
            gemini_model = genai.GenerativeModel("gemini-2.5-flash") 
            response = gemini_model.generate_content(prompt)
            return response.text
            
        elif provider == "Ollama":
            # Fallback to local
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
            
    except Exception as e:
        return f"Error querying {provider}: {str(e)}"
    
    return "Error: Invalid provider selected."