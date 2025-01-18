from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores.faiss import FAISS
import os
import config
from dotenv import load_dotenv



api_key = config.api_key



embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)

model = ChatMistralAI(mistral_api_key=api_key)



# Load the FAISS index with dangerous deserialization allowed (only if the source is trusted)
vector = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Define a retriever interface
retriever = vector.as_retriever()


# Define prompt template
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# Create a retrieval chain to answer questions
document_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "Whats the buy-in? What happens if he lies?"})
print(response["answer"])