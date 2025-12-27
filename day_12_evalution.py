import os
import faiss
import pickle
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(
    api_key=os.getenv('GROQ_API_KEY'),
    base_url= "https://api.groq.com/openai/v1"
)

VECTOR_DIR="vectorstore"
INDEX_PATH = os.path.join(VECTOR_DIR,'index.faiss')
DOCS_PATH = os.path.join(VECTOR_DIR,'docs.pkl')


index = faiss.read_index(INDEX_PATH)

with open(DOCS_PATH,'rb') as f:
    documents = pickle.load(f)
    
embed_model = SentenceTransformer("all-MiniLM-L6-v2",device="cpu")
    
tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(doc['text'] for doc in documents)

def hybrid_retrieval(query,top_k=5):
    query_vector = embed_model.encode([query])
    distances,indices = index.search(query_vector,8)
    
    vector_scores = {
        idx : 1/(1+dist)
        for dist, idx in zip(distances[0],indices[0])
    }
    
    query_tfidf = tfidf.transform([query])
    keyword_socres = cosine_similarity(
        tfidf_matrix,query_tfidf
    ).ravel()
    
    ALPHA, BETA = 0.6,0.4
    
    hybrid_scores = {
        i: ALPHA * vector_scores.get(i,0) + BETA * keyword_socres[i]
        for i in range(len(documents))
        
    }
    
    ranked_indices = sorted(
        hybrid_scores,
        key = hybrid_scores.get,
        reverse= True
    )
    
    return ranked_indices[:top_k]


def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & set(relevant))/max(len(relevant),1)

def precision_at_k(retrieved,relevant,k):
    return len(set(retrieved[:k]) & set(relevant))/k

def judge_faithfulness(answer,context):
    prompt = f"""
Answer:
{answer}

Context:
{context}

Is the answer fully supported by the context?
Reply ONLY YES or NO.
    """
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}],
        temperature=0        
    )
    
    return "YES" in r.choices[0].message.content.upper()

def judge_completeness(question,answer):
    prompt = f"""
Questions:
{question}

Answer:
{answer}

Does the answer fully address the question?
Reply ONLY YES or NO.
    """
    
    r =  client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return "YES" in r.choices[0].message.content.upper()

def final_rag_score(recall,precision,faithfullness,completeness):
    return(
        0.4* recall +
        0.2 * precision +
        0.2 * faithfullness +
        0.2 * completeness
    )
    
    
evaluation_set = [
    {
        "question": "What is Retrieval-Augmented Generation (RAG), and why is it critical for reducing hallucinations in modern generative AI systems?",
        "relevant_indices": [87, 88]
    },
    {
        "question": "Explain the difference between sparse retrieval, dense retrieval, and hybrid retrieval, and describe when hybrid retrieval is preferred in production RAG systems.",
        "relevant_indices": [102, 103, 104]
    }
]


print("\n Running RAG Evaluation...\n")

for item in evaluation_set:
    question = item['question']
    relevant = item['relevant_indices']
    
    retrieved = hybrid_retrieval(question,top_k=5)
    
    context = "\n\n".join(documents[i]['text'] for i in retrieved)
    
    answer_prompt = f"""
Use ONLY the context below to answer.

Context:
{context}

Question:
{question}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content

    recall = recall_at_k(retrieved, relevant, 5)
    precision = precision_at_k(retrieved, relevant, 5)
    faith = judge_faithfulness(answer, context)
    complete = judge_completeness(question, answer)

    score = final_rag_score(
        recall,
        precision,
        int(faith),
        int(complete)
    )

    print(f"📘 Question: {question}")
    print(f"Recall@5: {recall:.2f}")
    print(f"Precision@5: {precision:.2f}")
    print(f"Faithfulness: {faith}")
    print(f"Completeness: {complete}")
    print(f"FINAL RAG SCORE: {score:.2f}")
    print("-" * 50)
    