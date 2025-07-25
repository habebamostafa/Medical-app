import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import pandas as pd

# Initialize embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
@st.cache_resource
def load_data():
    try:
        train_data = pd.read_csv("data.csv")
        
        seen_drugs = set()
        documents = []
        
        for example in train_data:
            drug = example['drugName']
            if drug not in seen_drugs:
                condition = example['condition']
                review = example['review']
                rating = example['rating']
                text = f"Drug: {drug}\nCondition: {condition}\nRating: {rating}/10\nReview: {review}"
                documents.append(Document(page_content=text))
                seen_drugs.add(drug)
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        
        return chunks, seen_drugs
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return [], set()

chunks, seen_drugs = load_data()

# Initialize LLM
@st.cache_resource
def init_llm():
    return ChatOllama(
        model="deepseek-r1:1.5b",
        base_url="https://stupid-buses-draw.loca.lt"
    )

llm = init_llm()

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You are a highly qualified medical doctor specializing in pharmacology.
Your task is to assist patients by analyzing symptoms and suggesting treatments,
using only the provided medical context.

Guidelines:
1. Only recommend medications found in the context
2. Mention serious side effects as warnings
3. For severe symptoms, recommend doctor consultation
4. Never provide dosage or frequency advice
5. Use professional medical tone

Response format:
- **Patient Query:** Rephrased question
- **Recommendations:** Treatment suggestions
- **Warnings:** Any safety concerns"""),

    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Patient: {input}\n\nMedical Records:\n{context}")
])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="input")

# Helper functions
def get_relevant_chunks(question, k=5):
    try:
        question_embedding = embedder.encode(question, convert_to_tensor=True)
        chunk_texts = [c.page_content for c in chunks if hasattr(c, 'page_content')]
        chunk_embeddings = embedder.encode(chunk_texts, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]
        top_k_idx = scores.argsort(descending=True)[:k]
        return [chunks[i] for i in top_k_idx]
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def clean_response(response_text):
    return re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

def detect_drug(query):
    query_lower = query.lower()
    for drug in seen_drugs:
        if drug.lower() in query_lower:
            return drug
    return None

def ask_question(question, k=3):
    try:
        relevant_chunks = get_relevant_chunks(question, k)
        if not relevant_chunks:
            return "No information found about this medication."
            
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        
        memory.chat_memory.add_user_message(question)
        chain = create_stuff_documents_chain(llm, prompt)
        
        result = chain.invoke({
            "input": question,
            "context": context,
            "chat_history": memory.chat_memory.messages
        })
        
        cleaned = clean_response(result)
        memory.chat_memory.add_ai_message(cleaned)
        return cleaned
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "Couldn't process your question. Please try again."

# Streamlit UI
st.title("🤖 Medical Assistant")
st.write("Ask about medications and treatments")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Your medical question?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.spinner("Analyzing..."):
        response = ask_question(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})