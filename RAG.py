from sentence_transformers import SentenceTransformer
import faiss 
import ollama
import numpy as np 

class RAGPipeline:
    def __init__(self, model_name="all-MiniLM-L6-v2", llm_model="llama3.1:8b"):
      self.embedder=SentenceTransformer(model_name)
      self.index=None
      self.docs=[]
      self.llm_model=llm_model
    
    def build_index(self, documents):
       self.docs=documents
       vectors=self.embedder.encode(documents , convert_to_numpy=True)
       dim=vectors.shape[1]

       self.index=faiss.IndexFlatL2(dim)
       self.index.add(vectors)
    
    def query(self, question:str, k=5 ):
       if self.index is None:
            raise ValueError("Index not built yet. Call build_index first.")
       
       q_vec=self.embedder.encode([question], convert_to_numpy=True)
       distances ,indices=self.index.search(q_vec,k=k)

       hits=[(self.docs[i], float(distances[0][j])) for j, i in enumerate(indices[0])]
       return hits
    
    def answer(self, question: str, k=5):
         hits = self.query(question, k)
         context = "\n\n".join([f"Doc {i+1}: {doc}" for i, (doc, _) in enumerate(hits)])
         prompt = f"""You are a research assistant. 
         Use the following abstracts to answer the question concisely, 
         and cite supporting documents as (Doc 1, Doc 2, ...).

         Question: {question}

         Context:
         {context}

         Answer:"""

         resp = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}]
         )

         # ✅ Safe extraction
         if isinstance(resp, dict) and "message" in resp:
            answer = resp["message"]["content"]
         elif isinstance(resp, dict) and "content" in resp:
            answer = resp["content"]
         else:
            answer = str(resp)

         return answer.strip(), hits
