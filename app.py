

import os
import tempfile
import requests
from typing import List

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# HuggingFace Inference API Embeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from langchain_community.vectorstores import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

from pinecone import Pinecone as PineconeClient

load_dotenv()

# --- Configuration ---
API_TOKEN = "3b3b7f8e0cb19ee38fcc3d4874a8df6dadcdbfec21b7bbe39a73407e2a7af8a0"
PINECONE_INDEX_NAME = "hackrx-index"

# Hugging Face Token
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACE_API_TOKEN must be set in environment variables.")

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=hf_token,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- Authentication ---
auth_scheme = HTTPBearer()

# --- Pinecone ---
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
if not pinecone_api_key or not pinecone_env:
    raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT must be set in environment variables.")

pc = PineconeClient(api_key=pinecone_api_key)

# --- Gemini LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.2,
    convert_system_message_to_human=True,
)

app = FastAPI(
    title="HackRx 6.0 Intelligent Query-Retrieval System",
    description="API that processes a document URL and answers questions using Pinecone and Gemini.",
)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# --- Token Verification ---
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing authentication token")
    return credentials

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def run_query(request: QueryRequest):
    document_url = request.documents
    questions = request.questions
    tmp_file_path = None
    namespace = None

    try:
        # Step 1: Download the PDF
        response = requests.get(document_url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        # Step 2: Load & Split the PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)

        # Step 3: Make a unique namespace
        namespace = os.path.basename(tmp_file_path).replace(".pdf", "")

        # Step 4: Store to Pinecone
        vectorstore = Pinecone.from_documents(
            docs, embeddings, index_name=PINECONE_INDEX_NAME, namespace=namespace
        )
        retriever = vectorstore.as_retriever()

        # Step 5: QA Chain with Gemini
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
        )

        # Step 6: Answer questions
        processed_answers = []
        for question in questions:
            result = qa_chain.invoke({"query": question})
            processed_answers.append(result.get("result", "No answer found."))

        return QueryResponse(answers=processed_answers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error occurred: {str(e)}")

    finally:
        # Clean up the temp file and Pinecone namespace
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        if namespace:
            try:
                index = pc.Index(PINECONE_INDEX_NAME)
                index.delete(delete_all=True, namespace=namespace)
            except Exception as e:
                print(f"Error cleaning Pinecone namespace {namespace}: {e}")

@app.get("/")
def read_root():
    return {"message": "HackRx 6.0 Pinecone & Gemini solution is running."}
