Bajaj HackRx – Intelligent Query Retrieval System
Overview

This project was built as part of Bajaj HackRx 6.0. The goal was to design an LLM-powered intelligent query retrieval system that can process domain-specific documents (Insurance, Legal, HR, Compliance) and answer queries in a structured and explainable manner.

Features

Upload and process documents (PDF/DOCX/Emails).

Generate embeddings and store them in a vector database.

Perform semantic search and clause retrieval.

Query the system and get structured answers.

Focus on accuracy, explainability, and token efficiency.

Tech Stack

Backend: FastAPI

LLM: GPT-4 via Together API

Vector Database: Pinecone

Database: PostgreSQL

Embeddings: Sentence Transformers / LangChain

Project Structure

main.py → FastAPI backend and API endpoints

functions.py → Core logic for document processing and query handling

requirements.txt → Dependencies

Setup

Clone the repository.

Install dependencies:

pip install -r requirements.txt


Set up environment variables in a .env file:

TOGETHER_API_KEY=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

PINECONE_API_KEY=pcsk_9WA6p_DXRCfJKPY1Cbw4FfYnH6P2Bj2yf7KFDTnBGoE6bpmXDwxTY6XbxKwRnRRmVVMSc


Run the backend:

uvicorn main:app --reload

Outcome

The system successfully demonstrated:

Document processing and clause retrieval.

Query answering with structured JSON responses.
