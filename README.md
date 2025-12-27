# RAG-Based LLM Application

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system to improve answer accuracy and reduce hallucinations for document-based Q&A.

## Key Features
- Document ingestion and chunking
- Vector-based semantic search
- Context-aware LLM responses
- Prompt templating and retrieval optimization

## Architecture
1. Documents are split into semantic chunks  
2. Embeddings are generated and stored in a vector database  
3. Relevant context is retrieved via similarity search  
4. LLM generates grounded responses using retrieved context  

## Tech Stack
- Python
- LangChain
- Vector Database (FAISS / Chroma)
- LLM APIs (OpenAI-style / Groq)
