import streamlit as st
import pandas as pd
from Arxiv import fetch_papers
from embeddings import EmbedCluster
from summarizer import batch_summary
from summarizer import summarize_cluster

from RAG import RAGPipeline

st.set_page_config(page_title="Research Copilot", layout="wide")

st.title("📚 Research Copilot")
st.caption("Cluster papers + Summarize + Query-Aware Literature Review")

with st.sidebar:
    st.header("Settings")
    query = st.text_input("topic/query", value="graph neural network drug discovery")
    max_paper = st.slider("Max Result", 10, 100, 40, step=5)
    cluster = st.slider("No. of cluster", 2, 10, 5)
    summary_style = st.radio("Summary style", ["Bullets", "Paragraph"])
    go = st.button("run")

if go:
    # Phase 1 + 2
    with st.spinner("Fetching papers..."):
        papers = fetch_papers(query=query, max_paper=max_paper)
    if not papers:
        st.error("No results. Try a broader query.")
        st.stop()

    texts = [p["summary"] for p in papers]
    summaries = batch_summary(texts)
    for i, p in enumerate(papers):
        p["short_summary"] = summaries[i]

    # Build RAG index once
    rag = RAGPipeline()
    rag.build_index(texts)

    # Tabs: Exploration + Q&A
    tab1, tab2 = st.tabs(["📑 Clusters & Summaries", "❓ Research Q&A"])

    with tab1:
        df = pd.DataFrame(papers)
        st.subheader("Clustered View")
        st.dataframe(df[["title", "published", "pdf_url", "short_summary"]],
                     use_container_width=True, hide_index=True)
        
        with st.spinner("Clustering abstracts..."):
         texts = [p["summary"] for p in papers]
         ec = EmbedCluster()
         ec.fit(texts, papers)
         labels, _ = ec.kmeans(k=cluster)

        for i, p in enumerate(papers):
          p["cluster"] = int(labels[i])

    # Cluster-level summaries
        st.subheader("📝 Draft Literature Review")
        review_draft = []
        for c in sorted(set(labels)):
            st.markdown(f"## Theme {c+1}")
            subset = [p for p in papers if p["cluster"] == c]

            # Combine abstracts for cluster summary
            cluster_texts = [p["summary"] for p in subset]
            cluster_summary = summarize_cluster(cluster_texts, style=summary_style)

            st.write(cluster_summary)
            review_draft.append(f"### Theme {c+1}\n{cluster_summary}\n")

            st.markdown("**Key Papers:**")
            for p in subset[:3]:
              st.markdown(f"- {p['title']} ({p['published'].year}) – {', '.join(p['authors'][:3])} et al.")

    with tab2:
        
        st.subheader("Ask a Research Question")
        qa_q = st.text_input("Enter your question")
        top_k = st.slider("Top papers to use", 1, 10, 5)

        if st.button("Get Answer"):
            if not qa_q.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    with st.spinner("Retrieving and answering..."):
                        
                        st.write("DEBUG: calling rag.answer()...")
                        try:
                            result = rag.answer(qa_q, k=top_k)
                            st.write("DEBUG: rag.answer() returned:", result)
                            if result is None:
                                answer, hits = "", []
                            else:
                                answer, hits = result
                        except Exception as e:
                            st.error(f"RAG error: {e}")
                            answer, hits = "", []


                except Exception as e:
                    st.error(f"❌ Error while generating answer: {e}")


