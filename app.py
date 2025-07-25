import streamlit as st
from langchain_community.llms import HuggingFaceHub
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Constants
MAX_RESPONSE_TIME = 25  # seconds
TIMEOUT_BUFFER = 5  # seconds

# App configuration
st.set_page_config(
    page_title="💊 AI Medication Assistant",
    page_icon="💊",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Model loading with timeout handling
@st.cache_resource(show_spinner="Initializing AI models...")
def load_models():
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        llm = HuggingFaceHub(
            repo_id="deepseek-ai/deepseek-llm-7b-chat",
            model_kwargs={
                "temperature": 0.3,
                "max_new_tokens": 512,
                "timeout": 15  # More conservative timeout
            },
            huggingfacehub_api_token=st.secrets["huggingfacehub_api_token"]
        )
        return embedder, llm
    except Exception as e:
        st.error(f"Model initialization failed: {str(e)}")
        st.stop()

# Data loading with fallback
@st.cache_resource(show_spinner="Loading medication data...")
def load_data():
    try:
        # Try multiple data sources
        try:
            train_data = pd.read_csv("data.csv")
        except:
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
            documents.append({
                "text": text,
                "drug": row['drugName']
            })
        
        return documents, set(train_data['drugName'].unique())
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return [], set()

# Thread-safe response generation
def generate_response(query, context, llm):
    def _generate():
        try:
            prompt = f"""Answer this medication question based on the context:
            
            Context:
            {context}
            
            Question: {query}
            
            Provide:
            1. Primary Uses
            2. Effectiveness
            3. Side Effects
            4. Safety Notes
            
            Rules:
            - Never recommend dosages
            - Say "Not in database" for unknown drugs"""
            
            return llm(prompt)
        except Exception as e:
            raise RuntimeError(str(e))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_generate)
        try:
            return future.result(timeout=MAX_RESPONSE_TIME-TIMEOUT_BUFFER)
        except FutureTimeoutError:
            future.cancel()
            raise TimeoutError("Response timed out")

# Main processing function
def process_query(user_query, documents, embedder, llm):
    status = st.empty()
    start_time = time.time()
    
    try:
        # Update status with elapsed time
        def update_status():
            elapsed = int(time.time() - start_time)
            status.markdown(f"🔍 Analyzing... ({elapsed}s)")
        
        # Retrieve context
        update_status()
        query_embed = embedder.encode(user_query)
        doc_embeds = embedder.encode([d["text"] for d in documents])
        scores = util.pytorch_cos_sim(query_embed, doc_embeds)[0]
        top_indices = scores.topk(min(3, len(documents))).indices
        context = '\n\n'.join([documents[i]["text"] for i in top_indices])
        
        # Generate response
        update_status()
        response = generate_response(user_query, context, llm)
        
        # Post-processing
        response = re.sub(r"(?i)(dosage|take \d+ mg)", "[Consult your doctor]", response)
        return response
        
    except TimeoutError:
        return "⚠️ Response timed out. Please try again with a more specific question."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
    finally:
        status.empty()

# --- Main App ---
st.title("💊 AI Medication Assistant")

# Initialize components
embedder, llm = load_models()
documents, drug_set = load_data()

# Sidebar
with st.sidebar:
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.write(f"📊 {len(drug_set)} medications loaded")

# Chat interface
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about a medication..."):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        response = process_query(prompt, documents, embedder, llm)
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
