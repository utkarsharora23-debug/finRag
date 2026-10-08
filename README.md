# finRAG

A finance question-answering chatbot built with RAG (Retrieval Augmented Generation). It answers questions only from the finance PDFs you give it, and shows which file and page each answer came from.

## How it works

PDFs -> split into 500 character chunks -> embeddings (all-MiniLM-L6-v2) -> FAISS index -> top 3 chunks for a question -> Llama model on Groq writes the answer from those chunks.

## Tech stack

Python, LangChain, FAISS, HuggingFace sentence-transformers, Groq (Llama), Streamlit

## Files

| File | What it does |
|---|---|
| `create_memory_for_llm.py` | Reads PDFs from `data/`, builds the FAISS index in `vectorstore/` |
| `rag.py` | Loads the index and the LLM, and answers a question using the top 3 chunks |
| `finrag.py` | Streamlit chat app |
| `connect_memory_with_llm.py` | Same thing in the terminal, for quick testing |

## Setup

1. Create and activate a virtual environment

   ```
   python -m venv venv
   venv\Scripts\activate
   ```
   On Mac/Linux use `source venv/bin/activate`.

2. Install packages

   ```
   pip install -r requirements.txt
   ```

3. Get a free API key from https://console.groq.com, then copy `.env.example` to `.env` and paste the key in it.

4. Put your finance PDFs in the `data/` folder. Ideas: RBI or SEBI investor awareness booklets, company annual reports, a personal finance book, mutual fund guides.

5. Build the index (run again whenever you change the PDFs)

   ```
   python create_memory_for_llm.py
   ```

6. Start the app

   ```
   streamlit run finrag.py
   ```

## Notes

- The model name is set in `rag.py`. If Groq retires it, set `GROQ_MODEL` in your `.env` to another model from the Groq models list.
- The index is loaded with `allow_dangerous_deserialization=True`. This is fine because you build the index yourself. Never load an index file from someone else.
- Answers are general information, not financial advice.
- The chatbot treats each question separately. It does not remember earlier messages.
