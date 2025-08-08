from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os, json, traceback, requests, re
from dotenv import load_dotenv

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:10000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Request body model
class RunInput(BaseModel):
    documents: str  # URL to the PDF
    questions: List[str]


@app.post("/api/v1/hackrx/run")
async def run_hackrx(data: RunInput):
    try:
        # Download PDF from blob URL
        os.makedirs("downloads", exist_ok=True)
        pdf_path = "downloads/temp_policy.pdf"
        pdf_response = requests.get(data.documents)
        with open(pdf_path, "wb") as f:
            f.write(pdf_response.content)

        # Load and split PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Embed entire question block
        query_text = " ".join(data.questions)
        query_emb = embedding_model.encode(query_text)

        # Score each chunk using cosine similarity
        scored = []
        for chunk in chunks:
            content = chunk.page_content
            emb = embedding_model.encode(content)
            score = 1 - np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            scored.append((1 - score, content))  # higher is better

        # Take top 5 relevant chunks
        top_chunks = sorted(scored, reverse=True)[:5]
        context = "\n\n".join(chunk for _, chunk in top_chunks)

        # === ORIGINAL PROMPT PRESERVED ===
        prompt = f"""
You are an expert insurance policy analyst.

Your task is to analyze a customer query (which may be in informal or shorthand format) and respond with a short, clear, and accurate answer **based only on the provided insurance policy clauses**.

You must follow this **exact JSON format**:

{{
  "answers": [
    "Answer to query 1",
    "Answer to query 2",
    "Answer to query 3",
    "...",
    "Answer to query N"
  ]
}}

Guidelines:
- Understand informal inputs like "46M, knee surgery, Pune, 3-month policy" as a customer asking "Is knee surgery covered for a 46-year-old male in Pune under a policy active for 3 months?"
- Your answer must be **short**, **fact-based**, and **decisive** when possible (e.g., "Yes, knee surgery is covered under the policy.")
- Avoid vague phrases like "may be", "could be", or "might not".
- If the policy clearly supports or excludes a clause, give a definite "Yes..." or "No..." answer.
- If the document does NOT contain relevant information, respond with "Information not available in the provided document."
- Always use professional tone and complete sentences.
- The count and order of answers must match the number of customer queries.

Customer Queries:
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(data.questions)])}

Relevant Clauses:
{context}
"""

        # Call LLM (Together API)
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0
            }
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"LLM Error: {response.text}")

        reply = response.json()["choices"][0]["message"]["content"]

        # Extract JSON answer block
        match = re.search(r'\{\s*"answers"\s*:\s*\[.*?\]\s*\}', reply, re.DOTALL)
        if not match:
            raise ValueError(f"No valid JSON answers found. LLM response:\n\n{reply}")
        structured = json.loads(match.group())

        return {"response": structured}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
