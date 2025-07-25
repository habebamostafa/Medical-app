import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# App configuration
st.set_page_config(
    page_title="💊 AI Medication Assistant",
    page_icon="💊",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Model loading with enhanced error handling
@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    try:
        # Validate HuggingFace token
        if "huggingfacehub_api_token" not in st.secrets:
            st.error("🔑 HuggingFace API token not found in secrets!")
            st.stop()
            
        # Initialize embedding model
        embedder = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device='cuda' if st.secrets.get("use_gpu", False) else 'cpu'
        )
        
        # Initialize LLM with optimized parameters
        llm = HuggingFaceHub(
            repo_id="deepseek-ai/deepseek-llm-7b-chat",
            huggingfacehub_api_token=st.secrets["huggingfacehub_api_token"],
            model_kwargs={
                "temperature": 0.3,
                "max_new_tokens": 512,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "do_sample": True
            }
        )
        
        # Verify models are functional
        with st.spinner("🧪 Testing models..."):
            test_embed = embedder.encode("test", convert_to_tensor=True)
            test_response = llm.invoke("What is 1+1?")
            
        return embedder, llm
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

# Data loading with comprehensive validation
@st.cache_resource(show_spinner="Loading medication data...")
def load_data():
    try:
        # Load and validate dataset
        try:
            df = pd.read_csv("data.csv")
        except FileNotFoundError:
            st.error("📄 data.csv file not found!")
            st.stop()
            
        required_columns = {'drugName', 'condition', 'review', 'rating'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            st.error(f"🚨 Missing columns: {missing}")
            st.stop()
            
        # Preprocess data
        documents = []
        drug_set = set()
        
        for _, row in df.iterrows():
            drug = str(row['drugName']).strip()
            if not drug or drug.lower() == 'nan':
                continue
                
            condition = str(row['condition']) if pd.notna(row['condition']) else "Not specified"
            review = str(row['review']) if pd.notna(row['review']) else "No review"
            rating = str(row['rating']) if pd.notna(row['rating']) else "No rating"
            
            text = f"""Drug: {drug}
Condition: {condition} 
Rating: {rating}/10
Review: {review}"""
            
            documents.append(Document(
                page_content=text,
                metadata={"drug": drug, "condition": condition}
            ))
            drug_set.add(drug)
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.split_documents(documents)
        
        return chunks, drug_set
        
    except Exception as e:
        st.error(f"📊 Data processing error: {str(e)}")
        st.stop()

# System prompt with enhanced safety
MEDICAL_PROMPT = """You are a certified medical information assistant. Provide:

1. **Primary Uses**: Approved conditions and off-label uses
2. **Effectiveness**: Summary of patient experiences
3. **Side Effects**: Common (≥1%) and serious adverse effects
4. **Safety Notes**: Contraindications and black box warnings

**Rules**:
- Never prescribe or recommend dosages
- State "Not in database" for unknown drugs
- Highlight FDA approval status
- Cite sources when possible
- Disclose limitations of patient reviews"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", MEDICAL_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="input"
)

# Semantic search with score threshold
def retrieve_relevant_info(query, chunks, k=5, min_score=0.3):
    try:
        query_embed = embedder.encode(query, convert_to_tensor=True)
        chunk_texts = [c.page_content for c in chunks]
        chunk_embeds = embedder.encode(chunk_texts, convert_to_tensor=True)
        
        scores = util.pytorch_cos_sim(query_embed, chunk_embeds)[0]
        top_results = []
        
        for idx, score in enumerate(scores):
            if score > min_score:
                top_results.append((chunks[idx], float(score)))
                
        top_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in top_results[:k]]
        
    except Exception as e:
        st.error(f"🔍 Search error: {str(e)}")
        return []

# Process queries with context
def generate_response(user_query):
    try:
        # Retrieve relevant context
        context_docs = retrieve_relevant_info(user_query, chunks)
        if not context_docs:
            return "No verified information found about this medication."
        
        # Create processing chain
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(document_chain, memory)
        
        # Generate response
        result = retrieval_chain.invoke({
            "input": user_query,
            "context": context_docs
        })
        
        # Clean and format response
        response = result['answer']
        cleaned = re.sub(r"(?i)(dosage|take \d+ mg)", "[Dosage redacted - consult your doctor]", response)
        return cleaned.strip()
        
    except Exception as e:
        st.error(f"🤖 Response generation failed: {str(e)}")
        return "I encountered an error processing your request."

# --- UI Components ---
st.title("💊 AI-Powered Medication Assistant")
st.caption("Get accurate, evidence-based information about medications")

# Initialize models and data
embedder, llm = load_models()
chunks, drug_set = load_data()

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    if st.button("🔄 Clear Conversation"):
        st.session_state.messages = []
        memory.clear()
    
    st.divider()
    st.subheader("Database Info")
    st.metric("Total Medications", len(drug_set))
    
    if st.checkbox("Show sample drugs"):
        st.write(list(sorted(drug_set))[:20])

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
        with st.spinner("Analyzing medication info..."):
            response = generate_response(user_input)
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
