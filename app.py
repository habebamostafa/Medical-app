import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import re
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from langchain_community.llms import HuggingFaceHub
from huggingface_hub import login
import torch
login(token=st.secrets["hugging_face_api_token"])
# Constants
# MAX_RESPONSE_TIME = 30  # seconds
# RETRY_ATTEMPTS = 2
# TIMEOUT_BUFFER = 5  # seconds

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

def load_models():
    try:
        # Embedder as before
        embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Check for token
        if "hugging_face_api_token" not in st.secrets:
            st.error("🔑 Hugging Face token not found in secrets!")
            st.stop()

        # Use local Flan-T5 model with token
        model_name = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=st.secrets["hugging_face_api_token"])
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=st.secrets["hugging_face_api_token"])
        st.success("✅ Model loaded!")
        def local_llm(prompt):
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True)
                result=tokenizer.decode(outputs[0], skip_special_tokens=True)
                print("results".result)
                return result

        return embedder, local_llm

    except Exception as e:
        st.error(f"❌ Model initialization failed: {str(e)}")
        st.stop()


# Robust data loading with progress tracking
# @st.cache_resource(show_spinner="Loading medication data...")
def load_data():
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Searching for data sources...")
        progress_bar.progress(10)
        
        # Try multiple data sources with priority
        data_sources = [
            "data.csv"  # Local fallback
        ]
        
        train_data = None
        for source in data_sources:
            try:
                status_text.text(f"Attempting to load from: {source}")
                train_data = pd.read_csv(source)
                progress_bar.progress(50)
                break
            except Exception as e:
                st.warning(f"Failed to load from {source}: {str(e)}")
                continue
                
        if train_data is None:
            status_text.text("Using built-in sample data")
            st.warning("Using built-in sample data")
            train_data = pd.DataFrame({
                'drugName': ['Ibuprofen', 'Paracetamol', 'Aspirin'],
                'condition': ['Pain', 'Fever', 'Pain'],
                'review': ['Effective for headaches', 'Good for fever reduction', 'Helps with mild pain'],
                'rating': [8, 9, 7]
            })
            progress_bar.progress(60)

        # Validate data structure
        required_columns = {'drugName', 'condition', 'review', 'rating'}
        if not required_columns.issubset(train_data.columns):
            missing = required_columns - set(train_data.columns)
            raise ValueError(f"Missing required columns: {missing}")

        status_text.text("Processing documents...")
        progress_bar.progress(70)
        
        documents = []
        for _, row in train_data.iterrows():
            try:
                text = f"""Drug: {row['drugName']}
Condition: {row.get('condition', 'Not specified')}
Rating: {row.get('rating', 'No rating')}
Review: {row.get('review', 'No review')}"""
                documents.append(Document(
                    page_content=text,
                    metadata={"drug": row['drugName']}
                ))
            except Exception as e:
                st.warning(f"Skipping row due to error: {str(e)}")

        status_text.text("Splitting documents...")
        progress_bar.progress(80)
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        return chunks, set(train_data['drugName'].unique())
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Data loading error: {str(e)}")
        return [], set()

# System prompt with enhanced safety
MEDICAL_PROMPT = """You are a certified medical information assistant. Provide accurate information based on the context:

{context}

Your response must include:
1. **Primary Uses**: Approved conditions
2. **Effectiveness**: Patient-reported outcomes
3. **Side Effects**: Common and serious
4. **Safety Notes**: Important warnings

Strict Rules:
- NEVER recommend dosages (redact any dosage info)
- Clearly state when information is not available
- Highlight FDA approval status if known
- Maintain professional tone
- Disclose limitations of the information"""

# Thread-safe response generation with timeout
def generate_response_safe(query, context, llm):
    def _generate():
        try:
            prompt = f"""
            {MEDICAL_PROMPT}
            
            Context:
            {context}
            
            Question: {query}
            
            Provide a concise, professional response:"""
            print(llm(prompt))
            return llm(prompt)
        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}")

    # with ThreadPoolExecutor(max_workers=1) as executor:
        # future = executor.submit(_generate)
        # try:
        #     return future.result(timeout=MAX_RESPONSE_TIME-TIMEOUT_BUFFER)
        # except FutureTimeoutError:
        #     future.cancel()
        #     raise TimeoutError("Response generation timed out")

# Main response handler with retries and fallbacks
def get_response(user_query, chunks, llm, memory,embedder):
    print("👉 Starting response generation")
    # status = st.empty()
    # start_time = time.time()
    
    # def update_status():
    #     # elapsed = int(time.time() - start_time)
    #     status.markdown(f"🔍 Analyzing... ")
    
    try:
        # Retrieve context with progress updates
        print(",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,")
        query_embed = embedder.encode(user_query, convert_to_tensor=True)
        chunk_texts = [c.page_content for c in chunks]
        chunk_embeds = embedder.encode(chunk_texts, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embed, chunk_embeds)[0]
        top_indices = scores.topk(min(3, len(chunks))).indices
        context = '\n\n'.join([chunks[i].page_content for i in top_indices])
        print(context)
        # Generate response with retries
        last_error = None
        for attempt in range(RETRY_ATTEMPTS):
            # update_status()
            try:
                response = generate_response_safe(user_query, context, llm)
                
                # Post-processing
                response = re.sub(r"(?i)(dosage|take \d+ mg)", "[Consult your doctor]", response)
                memory.save_context({"input": user_query}, {"output": response})
                
                return response
                
            except TimeoutError as e:
                last_error = "Response timed out"
                if attempt < RETRY_ATTEMPTS-1:
                    time.sleep(1)  # Brief pause before retry
                    continue
            except Exception as e:
                last_error = str(e)
                break
                
        return f"⚠️ Could not generate response: {last_error or 'Unknown error'}. Please try again."
        
    except Exception as e:
        return f"🚨 An unexpected error occurred: {str(e)}"
    # finally:
    #     # status.empty()

# --- Main App ---
st.title("💊 AI-Powered Medication Assistant")

# Initialize components
try:
    embedder, llm = load_models()
    prompt = "Translate English to French: Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm.generate(**inputs)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    chunks, drug_set = load_data()
    
    if not chunks:
        st.warning("Warning: No medication data loaded. Some functionality may be limited.")
except Exception as e:
    st.error(f"Initialization failed: {str(e)}")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Clear Conversation", help="Start a new conversation"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()
    
    st.divider()
    st.subheader("Database Info")
    st.metric("Medications Available", len(drug_set))
    
    if st.checkbox("Show sample medications"):
        st.write(sorted(drug_set)[:10])

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask about a medication..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate and display response
    with st.chat_message("assistant"):
        response = get_response(
            user_input,
            chunks,
            llm,
            st.session_state.memory,
            embedder
        )
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
