import os
import pickle
import faiss
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# ENV + CLIENT
# ==============================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ==============================
# LOAD VECTORSTORE (DAY 9)
# ==============================

VECTOR_DIR = "vectorstore"
INDEX_PATH = os.path.join(VECTOR_DIR, "index.faiss")
DOCS_PATH = os.path.join(VECTOR_DIR, "docs.pkl")

index = faiss.read_index(INDEX_PATH)

with open(DOCS_PATH, "rb") as f:
    documents = pickle.load(f)

# ==============================
# MODELS
# ==============================

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================
# TF-IDF (DAY 10)
# ==============================

tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform([doc["text"] for doc in documents])

# ==============================
# STREAMLIT UI
# ==============================

st.set_page_config(page_title="Agentic RAG Chatbot")
st.title("🧠 Agentic RAG Book Chatbot (Day 11)")

# ==============================
# SESSION STATE
# ==============================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================
# DISPLAY HISTORY
# ==============================

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ==============================
# TOOLS
# ==============================

def hybrid_retrieve(query, top_k=5):
    """Hybrid search: FAISS + TF-IDF"""
    query_vector = embed_model.encode([query])
    distances, indices = index.search(query_vector, 8)

    vector_scores = {
        idx: 1 / (1 + dist)
        for dist, idx in zip(distances[0], indices[0])
    }

    query_tfidf = tfidf.transform([query])
    keyword_scores = cosine_similarity(tfidf_matrix, query_tfidf).ravel()

    ALPHA, BETA = 0.6, 0.4
    hybrid_scores = {
        i: ALPHA * vector_scores.get(i, 0) + BETA * keyword_scores[i]
        for i in range(len(documents))
    }

    top_indices = sorted(
        hybrid_scores,
        key=hybrid_scores.get,
        reverse=True
    )[:top_k]

    context = "\n\n".join(documents[i]["text"] for i in top_indices)
    sources = list({documents[i]["source"] for i in top_indices})

    return context, sources


def plan_steps(question):
    """Agent planner"""
    prompt = f"""
You are a planning agent.

Break the question into clear, ordered steps
needed to answer it completely.

Return steps as a numbered list.

Question:
{question}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    steps_text = response.choices[0].message.content
    steps = [
        line.strip("0123456789. ")
        for line in steps_text.split("\n")
        if line.strip()
    ]
    return steps


def verify_answer(answer):
    """Self-verification"""
    prompt = f"""
Does the following answer fully address the question?
Reply ONLY with YES or NO.

Answer:
{answer}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return "YES" in response.choices[0].message.content.upper()


# ==============================
# USER INPUT
# ==============================

query = st.chat_input("Ask a complex question from the book")

if query:
    st.chat_message("user").write(query)
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    # ==============================
    # AGENTIC LOOP
    # ==============================

    steps = plan_steps(query)
    st.markdown("### 🧠 Agent Plan")
    for i, step in enumerate(steps, 1):
        st.markdown(f"{i}. {step}")

    full_context = ""
    all_sources = []

    for step in steps:
        ctx, src = hybrid_retrieve(step)
        full_context += "\n\n" + ctx
        all_sources.extend(src)

    # ==============================
    # FINAL ANSWER
    # ==============================

    answer_prompt = f"""
You are an expert teacher.

Use ONLY the context below.
Do NOT guess or add external knowledge.
If something is missing, say so.

Context:
{full_context}

Question:
{query}

Provide a clear, structured answer.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0,
        max_tokens=1200
    )

    answer = response.choices[0].message.content

    # ==============================
    # VERIFICATION
    # ==============================

    is_complete = verify_answer(answer)

    if not is_complete:
        answer += "\n\n⚠️ The answer may be incomplete based on available context."

    # ==============================
    # DISPLAY
    # ==============================

    st.chat_message("assistant").write(answer)

    st.markdown("**Sources:**")
    for s in sorted(set(all_sources)):
        st.markdown(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
