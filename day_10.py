import os
import faiss
import pickle
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY")
    base_url="https://api.groq.com/openai/v1"
)

VECTOR_STORE_DIR = "vectorstore"
INDEX_PATH = os.path.join(VECTOR_STORE_DIR,'index.faiss')
DOCS_PATH = os.path.join(VECTOR_STORE_DIR,'docs.pkl')


index = faiss.read_index(INDEX_PATH)

with open(DOCS_PATH,'rb') as f:
    documents = pickle.load(f)
    
embed_model = SentenceTransformer('all-MiniLM-L6-v2')


# TF-IDF keyword search
tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2)
)

tfidf_matrix = tfidf.fit_transform(
    [doc['text'] for doc in documents]
)

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot")
st.title("RAG PDF Chatbot")

st.session_state.setdefault("messages",[])
st.session_state.setdefault("last_answer",[])
st.session_state.setdefault("last_context",[])
st.session_state.setdefault("last_sources",[])

# Display History
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Query Input
query = st.chat_input("Ask the question from the docuemnts")

if query:
    st.chat_message("user").write(query)
    st.session_state.messages.append(
        {'role':'user','content':query}
    )
    
    is_continue = query.strip().lower() == "continue"
    
    if is_continue:
        prompt = """
Continue the explanation later.
Do not repeat earlier content.

Previous Answer:
{st.session_state.last_answer}

context:
{st.session_state_last_context}

        """
        sources = st.session_state.last_sources
        
        
    else:
        query_vector = embed_model.encode([query])
        k = 8
        distances, indices = index.search(query_vector,k)
        
        vector_scores = {
            idx: 1/(1 + dist)
            for dist, idx in zip(distances[0],indices[0])
        }
        
        query_tfidf = tfidf.fit_transform([query])
        keyword_scores = (
            tfidf_matrix @ query_tfidf.T
        ).toarray().ravel()
        
        
        ALPHA = 0.6
        BETA = 0.4
        
        hybrid_scores = {}
        
        for i in range(len(documents)):
            v = vector_scores.get(i,0)
            kscore = keyword_scores[i]
            hybrid_scores[i] = ALPHA * v + BETA * kscore
            
            
        # Select top chunks
        top_inidces = sorted(
            hybrid_scores,
            key = hybrid_scores.get,
            reverse=True
        )[:5]
        
        context_chunks = []
        sources = []
        
        for idx in top_inidces:
            context_chunks.append(documents[idx]['text'])
            sources.append(documents[idx]['source'])
            
        context = "\n\n".join(context_chunks)
        
        st.session_state.last_context = context
        st.session_state.last_answer = ""
        st.session_state.last_sources = list(set(sources))
        
        
        # -------- Prompt --------
        prompt = f"""
You are an expert teacher.

Use ONLY the context below.
Do NOT guess or use prior knowledge.
If the answer is not found, say:
"Not found in the provided document."

Explain clearly using headings and bullet points.
If the explanation is long, stop and say:
"Type 'continue' to proceed."

Context:
{context}

Question:
{query}
"""

        # ==============================
        # LLM CALL
        # ==============================
        response = client.chat.completions.create(
            model = "llama-3.3.-70b-versatile",
            messages=[{'role':"user",'content':prompt}],
            temperature=0,
            max_tokens=900
        )
        
        answer = response.choices[0].message.content
        st.session_state.last_answer += "\n" + answer
        
        st.chat_message("assistant").write(answer)
        if sources:
            st.markdown("**Sources:**")
            for s in set(sources):
                st.markdown(f"-{s}")
                
        st.session_state.messages.append(
            {"role":"assistant","content":answer}
        )
            
        