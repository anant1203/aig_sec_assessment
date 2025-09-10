# Hybrid & Parent Document Retrieval

This repository contains experiments and implementations for building **hybrid search** and **parent–child document retrieval** pipelines using LangChain, FAISS, and semantic chunking. The goal is to improve retrieval performance on long documents by combining dense (vector) and sparse (keyword) search, while also preserving parent–child relationships between document sections and their semantic chunks.

---

## Project Structure

- **`hybrid_search.ipynb`**  
  Demonstrates hybrid retrieval by combining dense vector search (FAISS) with sparse keyword search (BM25).  
  Key features:
  - Indexing documents with OpenAI/Google embeddings.
  - Hybrid retriever setup with weighted scores.
  - Example queries comparing hybrid vs single-method results.
 
![Hybrid Search](./image/Solution%201.png)

- **`parent_document_retreiver.ipynb`**  
  Implements a **Parent Document Retriever** that chunks long documents into smaller child documents while keeping links back to their parent.  
  Key features:
  - Semantic chunking with `SemanticChunker` or `RecursiveCharacterTextSplitter`.
  - Parent–child indexing with FAISS and in-memory docstores.
  - Retrieval of child chunks along with their parent section context.

- **`WithoutVectorStore.ipynb`**  
  Baseline approach that performs retrieval **without a vector store**, useful for comparison.  
  Key features:
  - Rule-based/document-structured retrieval.
  - Demonstrates the limitations of non-embedding retrieval for semantic tasks.
 
  ![Without Vectoe Store](./image/Solution%202.png)

---

## Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/hybrid-parent-retrieval.git
   cd hybrid-parent-retrieval
   ```

2. **Install dependencies**  
   Create a virtual environment and install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   Typical dependencies include:
   - `langchain`
   - `langchain-community`
   - `langchain-experimental`
   - `faiss-cpu`
   - `rank-bm25`
   - `datasets`
   - `pandas`
   - `jupyter`

3. **Environment Variables**  
   Add your API keys in `.env` or environment variables:
   ```bash
   export OPENAI_API_KEY=your_key_here
   export GOOGLE_API_KEY=your_key_here
   ```

---

## Usage

Open Jupyter Lab/Notebook and run the notebooks in order depending on your experiment:

1. **Hybrid Search**  
   ```bash
   jupyter notebook hybrid_search.ipynb
   ```
   Explore hybrid retrieval across sample queries.

2. **Parent Document Retriever**  
   ```bash
   jupyter notebook parent_document_retreiver.ipynb
   ```
   Build and test parent–child retrievers with semantic chunking.

3. **Without Vector Store**  
   ```bash
   jupyter notebook WithoutVectorStore.ipynb
   ```
   Compare results to baseline approaches.

---

**Benefits**
- Returns few, **high-signal** contexts (full sections).
- Great for **structured reports** with clear section boundaries.
- Natural **traceability** from answer → parent section.

---

## Example Workflow

- Load SEC filings or other structured datasets (e.g., `eloukas/edgar-corpus`).
- Split each filing into **sections** (parent docs).
- Further chunk each section into **semantic child docs**.
- Index child docs in FAISS; maintain parent–child mapping in an in-memory store.
- Run retrieval with:
  - **Vector-only**
  - **Keyword-only**
  - **Hybrid**
  - **Parent Document Retriever**

---

## Acknowledgments
- LangChain community for `EnsembleRetriever` and `ParentDocumentRetriever` patterns.
- FAISS & Rank-BM25 for fast dense/sparse retrieval.
