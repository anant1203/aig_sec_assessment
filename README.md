# Hybrid & Parent–Child Retrieval – README

This repo/notebook demonstrates two practical retrieval patterns you can use for RAG over long, structured documents (e.g., SEC/EDGAR filings):

- **Solution 1 – Hybrid Retrieval**: Dense (FAISS) + Sparse (BM25) combined via `EnsembleRetriever`  
- **Solution 2 – Parent–Child Retrieval**: Chunk-and-lift with `ParentDocumentRetriever` to return full “parent” sections from “child” chunks

> Notebook: `hybrid_search.ipynb`  
> Diagrams: see below (Solution 1 and Solution 2).

---

## 1) Architecture

### Solution 1 – Hybrid Retrieval
![Solution 1](./image/Solution%201.png)

**Flow**
1. **Load Document(s)** and create a **hybrid vector store**.
2. Build two retrievers:
   - **Dense**: `FAISS` with your embedding model.
   - **Sparse**: `BM25Retriever` (token-based).
3. Combine them via **`EnsembleRetriever`** (weighted rank fusion).
4. Pass the retrieved contexts to your **LLM** for extraction or question answering.

**When to use**
- Queries can be semantic _or_ keyword-heavy.
- You want competitive recall with simple infrastructure.
- You do not need to reconstruct full sections/parents automatically.

---

### Solution 2 – Parent–Child Retrieval
![Solution 2](./image/Solution%202.png)

**Flow**
1. **Load Document(s)** and create **Parent docs** (e.g., one per section, file, or logical unit).  
2. **Split** parents into **Child chunks** (semantic/recursive splitter).  
3. Index **child chunks** in a vector store, but **return parent** documents on retrieval.  
4. Feed parent sections to your **LLM** for extraction or Q&A.

**When to use**
- You need **section-level** answers (e.g., _“Item 10 from AIG’s 2019 10‑K”_).
- You want fewer, richer contexts per query.
- You care about passage-to-section traceability (child → parent).

---

## 2) Quickstart

### Environment
```bash
# Python 3.10+ recommended
pip install -U "langchain>=0.2" langchain-community langchain-openai \
    faiss-cpu rank-bm25 datasets pandas numpy pydantic
# Optional: for Vertex AI or Google Generative AI embeddings
pip install -U google-generativeai google-cloud-aiplatform
```

> **Tip:** If you use OpenAI embeddings, set `OPENAI_API_KEY`.  
> For Vertex AI embeddings, authenticate with `gcloud auth application-default login`.

### Run
- Open `hybrid_search.ipynb` in Jupyter or VS Code.
- Execute cells top-to-bottom for **Solution 1** and **Solution 2** sections.
- Replace embedding/model choices as needed (OpenAI, Vertex AI, etc.).

---

## 3) Data (Example: SEC/EDGAR)
The notebook is data‑agnostic, but examples reference the public HF dataset **`eloukas/edgar-corpus`** (10‑K style sections like `section_1`, `section_7A`, `section_9B`, etc.).  
You can stream/filter by `cik`, `year`, and particular `section_*` columns before indexing.

> If you’re using very large files, prefer **streaming**, **on-disk caches** (e.g., JSONL/Parquet), and **chunked processing** to avoid RAM spikes.

---

## 4) How it Works

### A. Hybrid Retrieval (Solution 1)
**Key steps**
1. **Split** documents into chunks (e.g., `RecursiveCharacterTextSplitter`).
2. **Dense retriever**: index with FAISS (e.g., OpenAI or Vertex AI embeddings).
3. **Sparse retriever**: build a `BM25Retriever` from the same chunks.
4. **Combine**:
   ```python
   from langchain.retrievers import EnsembleRetriever
   dense = vectorstore.as_retriever(search_kwargs={"k": 6})
   sparse = bm25_retriever  # already a retriever
   hybrid = EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5])
   ```
5. **RAG**: call `hybrid.get_relevant_documents(query)` and send to your LLM.

**Tuning knobs**
- `chunk_size`, `chunk_overlap`
- `k` for dense/sparse retrievers
- `weights` for rank fusion in `EnsembleRetriever`

---

### B. Parent–Child Retrieval (Solution 2)
**Key steps**
1. **Parents**: create one doc per **logical section** (e.g., each `section_*` cell).  
   Include metadata: `filename`, `cik`, `year`, `section`, `parent_id` (e.g., `"{filename}#{section}"`).
2. **Children**: split each parent into chunks, attach `parent_id` to each child.
3. **Index**: only **children** go into FAISS (dense retrieval).
4. **Retrieve & Lift**: on a query, fetch child chunks, then **lift** to the **parent** doc(s).

**LangChain sketch**
```python
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

parent_store = InMemoryStore()  # stores full parents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
vectorstore  = FAISS.from_embeddings(...)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,        # where children live
    docstore=parent_store,          # where parents live
    child_splitter=child_splitter,
    search_kwargs={"k": 6},
)
```
> Depending on your LangChain version, `InMemoryStore` (not `InMemoryDocstore`) is the right type for `docstore`.

**Benefits**
- Returns few, **high-signal** contexts (full sections).
- Great for **structured reports** with clear section boundaries.
- Natural **traceability** from answer → parent section.

---

## 5) Common Pitfalls & Fixes

- **`ValidationError ... EnsembleRetriever ... Input should be an instance of Runnable`**  
  You likely passed a vector store instead of a retriever. Use `vectorstore.as_retriever(...)` for the dense side:
  ```python
  dense = vectorstore.as_retriever(search_kwargs={"k": 6})
  sparse = bm25_retriever
  hybrid = EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5])
  ```

- **Parent retriever error: `docstore` must be `BaseStore`**  
  On newer LangChain versions, use `from langchain.storage import InMemoryStore` (not `InMemoryDocstore`).

- **Google embeddings error: `unexpected model name format`**  
  Use a valid model id, e.g.:
  - Vertex AI: `text-embedding-004` (via `aiplatform`), or
  - Google Generative AI: `models/text-embedding-004`.
  Ensure the SDK matches the model provider.

- **FAISS dimension mismatch**  
  Probe dimension once and construct the FAISS index accordingly:
  ```python
  dim = len(embeddings.embed_query("dimension probe"))
  index = faiss.IndexFlatIP(dim)
  ```

- **OOM / large task warnings (Spark/JVM)**  
  - Prefer streaming + JSONL/Parquet staging.  
  - Increase driver/executor memory (`--driver-memory`, `--executor-memory`).  
  - Avoid collecting huge RDDs/DataFrames to the driver.

---

## 6) Evaluation (Optional)
For quick sanity checks:
- **Hit@k / Recall@k** over labeled queries → section ids.
- **MRR / nDCG** for ranking quality.
- **Answer Relevancy** (0–1) by a judge model or manual rubric.

Keep eval small and focused (e.g., 20–50 queries) to iterate on chunking & weights quickly.

---

## 7) Suggested Project Structure
```
.
├── hybrid_search.ipynb
├── Solution 1.png
├── Solution 2.png
└── README.md
```

---

## 8) Roadmap
- Add a lightweight **retrieval evaluator** notebook (Hit@k, MRR).
- Optional **re-ranker** (e.g., cross-encoder) atop hybrid retrieval.
- Cache embeddings & stores to disk for faster cold starts.

---

## 9) License
Provided as-is for internal experimentation. Add your preferred license if you plan to distribute.

---

## 10) Acknowledgments
- LangChain community for `EnsembleRetriever` and `ParentDocumentRetriever` patterns.
- FAISS & Rank-BM25 for fast dense/sparse retrieval.
