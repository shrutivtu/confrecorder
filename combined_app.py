from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pydantic import BaseModel
from typing import List
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv
import config
from lmnt.api import Speech
import os
import asyncio

api_key = config.api_key

# Define FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embeddings and model
embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
model = ChatMistralAI(mistral_api_key=api_key)

# Load FAISS index
vector = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Define retriever and chain
retriever = vector.as_retriever()
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")
document_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

class VectorStoreRequest(BaseModel):
    input: List[str]  # Accepts a list of strings

class VectorStoreResponse(BaseModel):
    message: str

class QueryRequest(BaseModel):
    input: List[str] 

class QueryResponse(BaseModel):
    answer: str

@app.post("/create_vector_store", response_model=VectorStoreResponse)
async def create_vector_store(request: VectorStoreRequest):
    try:
        # Concatenate the list of strings into a single string
        concatenated_text = " ".join(request.input)

        # Wrap the concatenated text into a Document object
        document = Document(page_content=concatenated_text)

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents([document])

        # Define the embedding model
        embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)

        # Create the vector store
        vector = FAISS.from_documents(documents, embeddings)

        # Save the vector store locally
        vector.save_local("faiss_index")

        return VectorStoreResponse(message="Created Vector Store successfully!")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_retrieval_chain(request: QueryRequest):
    try:
        # Concatenate the list of strings into a single string
        concatenated_input = " ".join(request.input)
        
        # Invoke the retrieval chain with the input question
        response = retrieval_chain.invoke({"input": concatenated_input})
        return QueryResponse(answer=response["answer"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/query_audio")
async def query_retrieval_chain(request: QueryRequest):
    try:
        # Concatenate the list of strings into a single string
        concatenated_input = " ".join(request.input)
        
        # Invoke the retrieval chain with the input question
        response = retrieval_chain.invoke({"input": concatenated_input})
        answer = response["answer"]

        # Convert text to speech
        os.environ['LMNT_API_KEY'] = config.lmnt_api_key

        async with Speech() as speech:
            synthesis = await speech.synthesize(answer, 'lily')
        
        audio_file_path = 'answer.mp3'
        with open(audio_file_path, 'wb') as f:
            f.write(synthesis['audio'])

        return FileResponse(audio_file_path, media_type='audio/mpeg', filename='answer.mp3')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


# Root endpoint for health check
@app.get("/")
async def health_check():
    return {"status": "OK"}
