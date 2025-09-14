from RAG import RAGPipeline

docs = [
    "Transformers are widely used in NLP tasks like translation and summarization.",
    "BLEU and ROUGE are common metrics for evaluating NLP models.",
    "Graph Neural Networks are applied in drug discovery using molecular graphs.",
]

rag = RAGPipeline()
rag.build_index(docs)

answer, hits = rag.answer("What metrics are used to evaluate NLP models?", k=2)
print("Answer:\n", answer)
print("\nRetrieved docs:")
for doc, dist in hits:
    print("-", doc[:100])
