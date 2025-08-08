from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX")

if not PINECONE_API_KEY or not INDEX_NAME:
    raise ValueError("Missing PINECONE_API_KEY or PINECONE_INDEX in environment variables.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ================================ #
# Load and Process PDF #
# ================================ #
def file_loader(file_path):
    """
    Loads a PDF, extracts text, chunks it, creates embeddings,
    and stores vectors in Pinecone.
    """
    print("Starting file_loader...")
    try:
        # Load PDF file
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Combine text from all pages
        extracted_text = " ".join([doc.page_content for doc in docs])

        # Split text into chunks
        split_docs = chunk_extracteddata(docs)

        # Generate embeddings
        embedding_function = embend_chunks()

        # Store in Pinecone
        retriever = vector_store_pinecone(embedding_function, split_docs)

        print(f"Indexed {len(split_docs)} chunks into Pinecone.")
        return retriever, extracted_text
    except Exception as e:
        print(f"[ERROR] file_loader failed: {str(e)}")
        raise

# ================================ #
# Split Documents to Chunks #
# ================================ #
def chunk_extracteddata(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)

# ================================ #
# HuggingFace Embeddings #
# ================================ #
def embend_chunks():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ================================ #
# Pinecone Vector Store #
# ================================ #
def vector_store_pinecone(embedding_function, split_docs):
    # Create index if it doesn't exist
    if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # MiniLM-L6-v2 has 384-dim vectors
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    # Initialize vector store
    vectorstore = PineconeVectorStore.from_documents(
        documents=split_docs,
        embedding=embedding_function,
        index_name=INDEX_NAME,
        namespace="default",
    )

    print("Data indexed into Pinecone.")
    return vectorstore.as_retriever()
