import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Use current absolute path as file dir
current_dir = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = os.path.join(current_dir, "rag_brochures")
DB_DIR = os.path.join(current_dir, "timber_vector_db")

# Load
loader = DirectoryLoader(FOLDER_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# Create and Save
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    persist_directory=DB_DIR
)

print(f"Success! Folder should now exist at: {DB_DIR}")