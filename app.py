import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage

# App configuration
st.set_page_config(
    page_title="💊 AI Medication Assistant",
    page_icon="💊",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Simplified model loading
@st.cache_resource(show_spinner="Initializing AI models...")
def load_models():
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        llm = HuggingFaceHub(
            repo_id="deepseek-ai/deepseek-llm-7b-chat",
            model_kwargs={
                "temperature": 0.3,
                "max_new_tokens": 512,
                "max_retries": 1,
                "timeout": 20
            },
            huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        )
        return embedder, llm
    except Exception as e:
        st.error(f"Model initialization failed: {str(e)}")
        st.stop()

# Robust data loading with fallback
@st.cache_resource(show_spinner="Loading medication data...")
def load_data():
    try:
        # Try multiple data sources
        data_sources = [
            "data.csv",
            "https://raw.githubusercontent.com/your_username/your_repo/main/data.csv"
        ]
        
        train_data = None
        for source in data_sources:
            try:
                train_data = pd.read_csv(source)
                break
            except:
                continue
                
        if train_data is None:
            st.warning("Using built-in sample data")
            train_data = pd.DataFrame({
                'drugName': ['Ibuprofen', 'Paracetamol', 'Aspirin'],
                'condition': ['Pain', 'Fever', 'Pain'],
                'review': ['Effective for headaches', 'Good for fever reduction', 'Helps with mild pain'],
                'rating': [8, 9, 7]
            })

        documents = []
        for _, row in train_data.iterrows():
            text = f"""Drug: {row['drugName']}
Condition: {row.get('condition', 'Not specified')}
Rating: {row.get('rating', 'No rating')}
Review: {row.get('review', 'No review')}"""
            documents.append(Document(
                page_content=text,
                metadata={"drug": row['drugName']}
            ))
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(documents), set(train_data['drugName'].unique())
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return [], set()

# System prompt
MEDICAL_PROMPT = """You are a medical information assistant. Provide:
1. Primary Uses
2. Effectiveness
3. Side Effects
4. Safety Notes

Rules:
- Never recommend dosages
- Say "Not in database" for unknown drugs
- Disclose limitations"""

# Simplified retrieval and generation
def get_response(user_query, chunks, llm, memory):
    try:
        # Retrieve relevant chunks
        query_embed = embedder.encode(user_query, convert_to_tensor=True)
        chunk_embeds = embedder.encode([c.page_content for c in chunks], convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embed, chunk_embeds)[0]
        context = [chunks[i] for i in scores.topk(3).indices]
        
        # Format prompt
        prompt = f"""
        {MEDICAL_PROMPT}
        
        Context:
        {'\n\n'.join([c.page_content for c in context])}
        
        Question: {user_query}
        
        Answer:"""
        
        # Get response with retry
        for _ in range(2):
            try:
                response = llm(prompt)
                memory.save_context(
                    {"input": user_query},
                    {"output": response}
                )
                return re.sub(r"(?i)(dosage|take \d+ mg)", "[Redacted - consult doctor]", response)
            except Exception:
                time.sleep(1)
                
        return "I couldn't generate a response. Please try again."
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "An error occurred while processing your request."

# --- Main App ---
st.title("💊 AI Medication Assistant")

# Initialize components
embedder, llm = load_models()
chunks, drug_set = load_data()

# Sidebar
with st.sidebar:
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()
    st.write(f"📊 {len(drug_set)} medications loaded")

# Chat interface
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about a medication..."):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            response = get_response(
                prompt,
                chunks,
                llm,
                st.session_state.memory
            )
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
