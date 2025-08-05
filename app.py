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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

from pinecone import Pinecone as PineconeClient, ServerlessSpec

load_dotenv() 

# --- Configuration ---
API_TOKEN = "3b3b7f8e0cb19ee38fcc3d4874a8df6dadcdbfec21b7bbe39a73407e2a7af8a0"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PINECONE_INDEX_NAME = "hackrx-index" # The name you gave your index in the Pinecone dashboard

# --- Authentication ---
auth_scheme = HTTPBearer()

# --- Initialize Services (once on startup) ---
print("Initializing Pinecone client...")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

if not pinecone_api_key or not pinecone_env:
    raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT must be set in the environment.")

pc = PineconeClient(api_key=pinecone_api_key)

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print("Initializing Google Gemini model...")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2, convert_system_message_to_human=True)
print("Models loaded successfully.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="HackRx 6.0 Intelligent Query-Retrieval System",
    description="An API that processes a document URL and answers questions using Pinecone."
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
        # 1. Download the PDF
        print(f"Downloading document from: {document_url}")
        response = requests.get(document_url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        
        # 2. Load and process the PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)

        # 3. Create a unique namespace for this document to keep it separate from others
        namespace = os.path.basename(tmp_file_path).replace(".pdf", "")

        print(f"Uploading document to Pinecone index '{PINECONE_INDEX_NAME}' with namespace '{namespace}'...")
        # This sends the document chunks to Pinecone, which creates the index in the cloud
        vectorstore = Pinecone.from_documents(
            docs, embeddings, index_name=PINECONE_INDEX_NAME, namespace=namespace
        )
        retriever = vectorstore.as_retriever()
        
        # 4. Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever, 
        )

        # 5. Answer questions
        processed_answers = []
        for question in questions:
            print(f"Answering question: '{question}'")
            result = qa_chain.invoke({"query": question})
            processed_answers.append(result.get("result", "No answer found."))
        
        return QueryResponse(answers=processed_answers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
    finally:
        # 6. Clean up resources
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            print(f"Cleaned up temporary file: {tmp_file_path}")
        if namespace:
            try:
                index = pc.Index(PINECONE_INDEX_NAME)
                index.delete(delete_all=True, namespace=namespace)
                print(f"Cleaned up Pinecone namespace: {namespace}")
            except Exception as e:
                print(f"Error cleaning up Pinecone namespace {namespace}: {e}")

@app.get("/")
def read_root():
    return {"message": "HackRx 6.0 Pinecone & Gemini solution is running."}
