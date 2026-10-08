import glob
import sys

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import DB_FAISS_PATH, get_embedding_model

DATA_PATH = "data/"

# Step 1: Load raw PDF files
if not glob.glob(DATA_PATH + "*.pdf"):
    sys.exit("No PDF files found in the data/ folder. Add some finance PDFs and run again.")

loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print("PDF pages loaded:", len(documents))

# Step 2: Create chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)
print("Text chunks created:", len(text_chunks))

# Step 3 and 4: Create embeddings and store them in FAISS
db = FAISS.from_documents(text_chunks, get_embedding_model())
db.save_local(DB_FAISS_PATH)
print("Vector store saved to", DB_FAISS_PATH)
