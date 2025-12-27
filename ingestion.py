import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# path
PDF_PATH = "data/Generative AI and Agentic AI.pdf"   # <-- update if needed
VECTOR_DIR = "vectorstore"
INDEX_PATH = os.path.join(VECTOR_DIR, "index.faiss")
DOCS_PATH = os.path.join(VECTOR_DIR, "docs.pkl")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# ==============================
# LOAD PDF (with metadata)
# ==============================

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

print(f"Loaded {len(pages)} pages")

# ==============================
# CHUNKING
# ==============================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")

# ==============================
# BUILD METADATA-AWARE DOCUMENTS
# ==============================

documents = []

for chunk in chunks:
    text = chunk.page_content.strip()
    page = chunk.metadata.get("page", "Unknown")

    documents.append({
        "text": text,
        "source": f"Page {page}"
    })

print("Sample document:")
print(documents[0])

# ==============================
# EMBEDDINGS
# ==============================

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embed_model.encode(
    [doc["text"] for doc in documents],
    show_progress_bar=True
)

dimension = embeddings.shape[1]

# ==============================
# FAISS INDEX
# ==============================

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ==============================
# SAVE EVERYTHING
# ==============================

os.makedirs(VECTOR_DIR, exist_ok=True)

faiss.write_index(index, INDEX_PATH)

with open(DOCS_PATH, "wb") as f:
    pickle.dump(documents, f)

print("✅ Day 9 ingestion completed successfully")