# 🎓 Research Copilot: Hybrid RAG & Visual Literature Review

**An AI-powered research assistant that aggregates Arxiv papers and local PDFs into a unified semantic knowledge base.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![RAG](https://img.shields.io/badge/Architecture-Hybrid%20RAG-green)
![LLM](https://img.shields.io/badge/AI-Gemini%20%7C%20Ollama-orange)

### 🚀 Overview
Research Copilot is a **Hybrid Retrieval-Augmented Generation (RAG)** application designed to streamline the academic review process. Unlike standard PDF chat bots, this tool bridges the gap between public research (Arxiv API) and private data (Local PDFs).

It uses **Unsupervised Learning (K-Means)** to cluster documents by theme, visualizes the research landscape using **PCA & Plotly**, and enables "Chat with Data" functionality using **Google Gemini** (Cloud) or **Ollama** (Local Privacy-Focused).

### ✨ Key Features

* **🗂️ Hybrid Data Ingestion:** Seamlessly merges live Arxiv search results with user-uploaded PDFs into a single vector index.
* **🧠 Dual-Engine Intelligence:**
    * **Cloud Mode:** Uses Google Gemini 1.5 Flash for high-speed, high-context reasoning.
    * **Local Mode:** Uses Ollama (Llama 3) for offline, privacy-centric analysis.
* **🗺️ Semantic Visualization:** Projects high-dimensional embeddings into a 2D interactive map (using PCA) to identify research clusters and outliers visually.
* **🤖 Automated Synthesis:** Generates thematic literature reviews for specific clusters on-demand, saving token costs.
* **💬 Context-Aware Chat:** A chat interface that cites sources (e.g., "Doc 1", "Upload") to ensure hallucination-free answers.

---

### 🛠️ Architecture

The application follows a modular pipeline:

1.  **Ingestion:**
    * **Arxiv:** Fetches metadata and abstracts via `arxiv` API.
    * **PDF:** Extracts text from local files via `pypdf`.
2.  **Embedding:** Converts text to 384-dimensional vectors using `all-MiniLM-L6-v2` (SentenceTransformers).
3.  **Clustering:**
    * Applies **K-Means** to group semantically similar papers.
    * Applies **PCA (Principal Component Analysis)** to reduce dimensions for visualization.
4.  **Storage:** Vectors are stored in a transient **FAISS** index for millisecond-level retrieval.
5.  **Generation:** Relevant context is retrieved and passed to the selected LLM (Gemini or Llama) for synthesis.

---

### 📦 Installation

**Prerequisites:**
* Python 3.9+
* [Ollama](https://ollama.com/) (Only if using Local Mode)

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/SpandanNagale/Literature-Review-creator.git](https://github.com/yourusername/research-copilot.git)
    cd research-copilot
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **(Optional) Setup Local LLM**
    If you plan to use the local mode, ensure Ollama is running and pull the model:
    ```bash
    ollama pull llama3.1:8b
    ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

---

### 🖥️ Usage Guide

**1. Choose Your Engine:**
* **Gemini (Cloud):** Select this in the sidebar. You will need a free API Key from [Google AI Studio](https://aistudio.google.com/).
* **Ollama (Local):** Select this if you have Ollama installed locally. No internet required.

**2. Load Data:**
* **Search Arxiv:** Enter a topic (e.g., "Generative Adversarial Networks") and hit Run.
* **Upload PDFs:** Drag and drop your own research papers.
* **Hybrid:** Do both! The app will find connections between your files and the web.

**3. Analyze:**
* **Map Tab:** Explore the semantic clusters. Hover over dots to see titles.
* **Review Tab:** Click "Synthesize Theme" to get a summary of a specific cluster.
* **Chat Tab:** Ask questions like *"How does the methodology in my uploaded PDF differ from the Arxiv papers?"*

---

### 📂 Project Structure

```text
research-copilot/
├── app.py              # Main Streamlit application entry point
├── llm_helper.py       # Handler for switching between Gemini and Ollama
├── embeddings.py       # Logic for embedding, K-Means, and PCA
├── pdf_loader.py       # Utility to parse and clean uploaded PDFs
├── Arxiv.py            # Wrapper for the Arxiv API
├── RAG.py              # Vector search and Retrieval logic
├── summarizer.py       # Prompts for summarization tasks
└── requirements.txt    # Project dependencies