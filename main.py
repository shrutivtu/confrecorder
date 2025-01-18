from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores.faiss import FAISS
import config

# Load API key
api_key = config.api_key

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

# Define FastAPI app
app = FastAPI()

# Define request and response models
class QueryRequest(BaseModel):
    input: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/query", response_model=QueryResponse)
async def query_retrieval_chain(request: QueryRequest):
    try:
        # Invoke the retrieval chain with the input question
        response = retrieval_chain.invoke({"input": request.input})
        return QueryResponse(answer=response["answer"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint for health check
@app.get("/")
async def health_check():
    return {"status": "OK"}
