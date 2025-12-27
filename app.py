import streamlit as st
import pandas as pd
import plotly.express as px

# --- CUSTOM MODULES ---
from Arxiv import fetch_papers
from embeddings import EmbedCluster
from summarizer import summarize_cluster
from RAG import RAGPipeline
from pdf_loader import parse_pdf  # ensure pdf_loader.py exists
# from llm_helper import query_llm # (Internal use only, imported by summarizer/RAG)

st.set_page_config(page_title="Research Copilot 3.0", layout="wide", page_icon="🎓")

# --- SESSION STATE INITIALIZATION ---
if "papers" not in st.session_state: st.session_state.papers = []
if "rag" not in st.session_state: st.session_state.rag = None
if "labels" not in st.session_state: st.session_state.labels = []
if "coords" not in st.session_state: st.session_state.coords = []
if "messages" not in st.session_state: st.session_state.messages = []
if "trigger_run" not in st.session_state: st.session_state.trigger_run = False
if "data_processed" not in st.session_state: st.session_state.data_processed = False

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("🎓 Research Copilot")
    st.caption("Hybrid RAG: Arxiv + Local PDFs")
    
    st.header("🧠 1. AI Engine")
    llm_provider = st.radio("Provider", ["Ollama (Local)", "Gemini (Cloud)"])
    
    api_key = None
    if llm_provider == "Gemini (Cloud)":
        api_key = st.text_input("Google API Key", type="password", help="Required for Gemini")
        if not api_key:
            st.warning("⚠️ API Key missing")
    
    # Global Config Dictionary
    llm_config = {
        "provider": "Gemini" if "Gemini" in llm_provider else "Ollama",
        "model": "gemini-2.5-flash" if "Gemini" in llm_provider else "gemma3:latest",
        "api_key": api_key
    }

    st.divider()
    
    st.header("📂 2. Data Sources")
    
    # Source A: Local Files
    uploaded_files = st.file_uploader("Upload PDFs (Optional)", type=["pdf"], accept_multiple_files=True)
    
    # Source B: Arxiv
    query = st.text_input("Arxiv Topic", value="graph neural network drug discovery")
    
    with st.expander("Advanced Search Settings"):
        max_paper = st.slider("Max Arxiv Results", 10, 100, 25, step=5)
        n_clusters = st.slider("Number of Clusters", 2, 8, 4)
        summary_style = st.radio("Summary Style", ["Bullets", "Paragraph"])

    st.divider()
    
    if st.button("🚀 Run Research Analysis", type="primary"):
        st.session_state.trigger_run = True

# --- CORE PIPELINE LOGIC ---
def run_pipeline():
    # Clear Chat History on new run
    st.session_state.messages = []
    
    with st.status("🤖 AI Agent Working...", expanded=True) as status:
        
        all_papers = []

        # 1. Process Local PDFs
        if uploaded_files:
            status.write(f"Processing {len(uploaded_files)} uploaded files...")
            for f in uploaded_files:
                parsed = parse_pdf(f)
                if parsed:
                    # Tag source for visualization
                    parsed["source"] = "Upload"
                    all_papers.append(parsed)

        # 2. Fetch Arxiv Papers
        if query.strip():
            status.write(f"Fetching papers from Arxiv for '{query}'...")
            arxiv_results = fetch_papers(query=query, max_paper=max_paper)
            for p in arxiv_results:
                p["source"] = "Arxiv"
            all_papers.extend(arxiv_results)

        # 3. Validation
        if not all_papers:
            status.update(label="No data found! Please upload a PDF or enter a query.", state="error")
            st.stop()
            
        # 4. Embed & Cluster
        status.write(f"Analyzing {len(all_papers)} documents...")
        texts = [p["summary"] for p in all_papers]
        
        ec = EmbedCluster()
        ec.fit(texts, all_papers)
        
        # Safe clustering: Ensure we don't ask for more clusters than papers
        actual_k = min(n_clusters, len(all_papers))
        labels, _ = ec.kmeans(k=actual_k)
        
        # Reduce dimensions for visualization
        coords = ec.reduce_dimensions()
        
        # Assign cluster labels back to papers
        for i, p in enumerate(all_papers):
            p["cluster"] = int(labels[i])
            # Default short summary is just title until AI generates one
            p["short_summary"] = p["title"]

        # 5. Build RAG Index
        status.write("Building Knowledge Base...")
        rag = RAGPipeline()
        rag.build_index(texts)

        # 6. Save State
        st.session_state.papers = all_papers
        st.session_state.rag = rag
        st.session_state.labels = labels
        st.session_state.coords = coords
        st.session_state.data_processed = True
        
        status.update(label="Research Complete!", state="complete", expanded=False)

# Trigger Handling
if st.session_state.trigger_run:
    run_pipeline()
    st.session_state.trigger_run = False

# --- UI RENDERING ---
if st.session_state.data_processed:
    papers = st.session_state.papers
    rag = st.session_state.rag
    labels = st.session_state.labels
    coords = st.session_state.coords

    # Dynamic Title
    display_title = f"📚 Analysis: {query}" if query else "📚 Analysis: Local Files"
    st.title(display_title)
    
    tab1, tab2, tab3 = st.tabs(["🗺️ Cluster Map", "📝 Literature Review", "🧠 Q&A Assistant"])

    # TAB 1: VISUALIZATION
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
             # Prepare plotting data
             plot_df = pd.DataFrame({
                 "x": coords[:, 0],
                 "y": coords[:, 1],
                 "title": [p["title"] for p in papers],
                 "cluster": [str(c) for c in labels],
                 "source": [p.get("source", "Arxiv") for p in papers],
                 "year": [str(p["published"])[:4] if p["published"] != "Local File" else "Local" for p in papers]
             })
             
             # Visualize with distinction between Uploads and Arxiv
             fig = px.scatter(
                 plot_df, x="x", y="y", color="cluster", symbol="source",
                 hover_data=["title", "year"],
                 title="Semantic Research Landscape",
                 template="plotly_dark",
                 size_max=12
             )
             st.plotly_chart(fig, use_container_width=True)
             
        with col2:
            st.metric("Total Documents", len(papers))
            st.metric("Clusters Found", len(set(labels)))
            st.markdown("### How to read this:")
            st.caption("• **Dots close together** are semantically similar.")
            st.caption("• **Colors** represent automated themes.")
            st.caption("• **Shapes** distinguish Arxiv vs. Uploads.")

    # TAB 2: SYNTHESIS
    with tab2:
        st.subheader("Automated Literature Review")
        st.info("💡 Click 'Generate Synthesis' to use the AI. This saves costs/time by not summarizing everything at once.")
        
        for c in sorted(set(labels)):
            subset = [p for p in papers if p["cluster"] == c]
            
            with st.expander(f"📌 Theme {c+1} ({len(subset)} documents)", expanded=False):
                col_a, col_b = st.columns([3, 1])
                
                with col_a:
                    btn_key = f"btn_sum_{c}"
                    if st.button(f"Generate Synthesis for Theme {c+1}", key=btn_key):
                        # API Key Check
                        if llm_config["provider"] == "Gemini" and not llm_config["api_key"]:
                            st.error("❌ Please enter a Google API Key in the sidebar.")
                        else:
                            with st.spinner("Synthesizing insights..."):
                                cluster_texts = [p["summary"] for p in subset]
                                # Pass config to summarizer
                                summary = summarize_cluster(cluster_texts, style=summary_style, config=llm_config)
                                st.success(summary)
                    
                    st.markdown("---")
                    st.markdown("**Documents in this theme:**")
                    for p in subset:
                        link = p['pdf_url'] if p['pdf_url'] != "#" else "Local Upload"
                        st.markdown(f"- **{p['title']}** [{link}]")
                
    # TAB 3: CHATBOT
    with tab3:
        st.subheader("Chat with your Knowledge Base")
        
        # Display History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat Input
        if prompt := st.chat_input("Ask a question about these papers..."):
            
            # API Key Check
            if llm_config["provider"] == "Gemini" and not llm_config["api_key"]:
                st.error("❌ Please enter a Google API Key in the sidebar.")
            else:
                # User Message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # AI Response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # Pass config to RAG
                            answer, hits = rag.answer(prompt, config=llm_config, k=4)
                            st.markdown(answer)
                            
                            with st.expander("View Sources"):
                                for doc_content, score in hits:
                                    st.caption(f"**Score: {score:.2f}** | ...{doc_content[:150]}...")
                            
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

else:
    # EMPTY STATE
    st.info("👈 Upload a PDF or enter a topic in the sidebar, then click 'Run Research Analysis'.")