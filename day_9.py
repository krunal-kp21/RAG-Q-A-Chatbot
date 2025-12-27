import os
import pickle
import faiss
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load environment variable
load_dotenv()

# Client setup
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


VECTOR_STORE_DIRS = "vectorstore"
INDEX_PATH = os.path.join(VECTOR_STORE_DIRS,'index.faiss')
DOCS_PATH = os.path.join(VECTOR_STORE_DIRS,'docs.pkl')

index = faiss.read_index(INDEX_PATH)

with open(DOCS_PATH,'rb') as f:
    documents = pickle.load(f)
    
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI

st.set_page_config(page_title="RAG Chatbot")
st.title("RAG PDF Chatbot")

st.session_state.setdefault("messages",[])
st.session_state.setdefault("last_context",[])
st.session_state.setdefault("last_answer",[])
st.session_state.setdefault("last_sources",[])


# show History
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

query = st.chat_input("ASk the question from the documents")


if query:
    st.chat_message("user").write(query)
    st.session_state.messages.append(
        {'role':'user','content':query}
    )
    
    is_continue = query.strip().lower() == "continue"
    
    if is_continue:
        prompt = f"""
Continue the explanation below.
Do NOT repeat previous content.

Previous answer:
{st.session_state.last_answer}

context:
{st.session_state.last_context}

          
        """
        sources = st.session_state.last_sources
        
    else:
        # ==============================
        # Retrieval
        # ==============================

        query_vector = embed_model.encode([query])
        k = 8
        distances, indices = index.search(query_vector, k)

        DISTANCE_THRESHOLD = 1.2
        context_chunks = []
        sources = []

        for dist, idx in zip(distances[0], indices[0]):
            if dist < DISTANCE_THRESHOLD:
                context_chunks.append(documents[idx]["text"])
                sources.append(documents[idx]["source"])

        if not context_chunks:
            context_chunks.append(documents[indices[0][0]]["text"])
            sources.append(documents[indices[0][0]]["source"])

        context = "\n\n".join(context_chunks)
        st.session_state.last_context = context
        st.session_state.last_sources = list(set(sources))

        prompt = f"""
You are an expert teacher.

Use ONLY the context below.
Do NOT guess or use prior knowledge.
If the answer is not found, say:
"Not found in the provided document."

Explain clearly with headings and bullet points.
If long, stop and say:
"Type 'continue' to proceed."

Context:
{context}

Question:
{query}
"""

    # ==============================
    # LLM Call
    # ==============================

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=900
    )

    answer = response.choices[0].message.content
    st.session_state.last_answer += "\n" + answer

    # ==============================
    # Display Answer + Sources
    # ==============================

    st.chat_message("assistant").write(answer)

    if sources:
        st.markdown("**Sources:**")
        for s in set(sources):
            st.markdown(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )   

