import os
import random
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "llama3")
PDF_FILE = os.getenv("PDF_FILE", "medic.pdf")

# Load PDF and split into chunks
loader = PyPDFLoader(PDF_FILE)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Generate embeddings and create FAISS index
embeddings = OllamaEmbeddings(model=MODEL_NAME)
vectorstore = FAISS.from_documents(docs, embeddings)

# Initialize the LLM
llm = Ollama(model=MODEL_NAME)

# Define prompt
prompt_template = """
Answer the question based on the following context:
Context:
---------
{context}

Chat History:
{chat_history}

Question: {input}
Answer (in one paragraph):
"""

prompt = PromptTemplate.from_template(prompt_template)

# Memory
memory = ConversationBufferMemory(return_messages=True)

# Function to get relevant chunks
def get_relevant_chunks(query, k=2):
    results = vectorstore.similarity_search(query, k=k)
    return results  # This always returns Document objects

# Function to clean model response
def clean_response(text):
    return text.replace("Answer:", "").replace("answer:", "").strip()

# Function to ask question with memory
def ask_question_with_memory(question, k=2):
    try:
        if not llm:
            return "System error: Model not loaded"

        question = str(question)[:300]

        relevant_chunks = get_relevant_chunks(question, k)

        # Safely extract text from Document objects
        context_text = "\n\n".join(
            getattr(doc, "page_content", str(doc)) for doc in relevant_chunks[:1]
        )

        # Add user question to memory
        memory.chat_memory.add_user_message(question)

        # Create the chain and generate answer
        chain = create_stuff_documents_chain(llm, prompt)
        result = chain.invoke({
            "input": question,
            "context": context_text,
            "chat_history": memory.chat_memory.messages[-2:]
        })

        # Clean and add response to memory
        response = clean_response(str(result))[:800]
        memory.chat_memory.add_ai_message(response)
        return response

    except Exception as e:
        return f"System error: {str(e)[:150]}"

# Test the question
print(ask_question_with_memory("What is the Bupropion medicine for?"))
