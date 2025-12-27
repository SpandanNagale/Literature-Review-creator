from sentence_transformers import SentenceTransformer
import faiss 
import numpy as np 
from llm_helper import query_llm

class RAGPipeline:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
      self.embedder = SentenceTransformer(model_name)
      self.index = None
      self.docs = []
    
    def build_index(self, documents):
       self.docs = documents
       vectors = self.embedder.encode(documents, convert_to_numpy=True)
       dim = vectors.shape[1]

       self.index = faiss.IndexFlatL2(dim)
       self.index.add(vectors)
    
    def query(self, question: str, k=5):
       if self.index is None:
            raise ValueError("Index not built yet.")
       
       q_vec = self.embedder.encode([question], convert_to_numpy=True)
       distances, indices = self.index.search(q_vec, k=k)

       hits = [(self.docs[i], float(distances[0][j])) for j, i in enumerate(indices[0])]
       return hits
    
    def answer(self, question: str, config: dict, k=5):
         hits = self.query(question, k)
         context = "\n\n".join([f"Doc {i+1}: {doc}" for i, (doc, _) in enumerate(hits)])
         
         prompt = f"""You are a research assistant. 
         Use the provided abstracts to answer the question.
         Cite sources as (Doc 1, Doc 2).
         
         Question: {question}
         
         Context:
         {context}
         """

         answer = query_llm(prompt, config)
         return answer, hits