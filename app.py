import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize Streamlit app
st.set_page_config(page_title="💊 Medication Information System", page_icon="💊")

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        llm = ChatOllama(base_url="http://localhost:11434", 
                        model="llama2",
                        temperature=0.3)
        return embedder, llm
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.stop()

embedder, llm = load_models()

# Load and preprocess data
@st.cache_resource
def load_data():
    try:
        train_data = pd.read_csv("data.csv")
        
        # Validate required columns
        required_columns = {'drugName', 'condition', 'review', 'rating'}
        if not required_columns.issubset(train_data.columns):
            missing = required_columns - set(train_data.columns)
            st.error(f"Missing required columns: {missing}")
            st.stop()
            
        # Process data into documents
        documents = []
        seen_drugs = set()
        
        for _, row in train_data.iterrows():
            drug = str(row['drugName']).strip()
            if not drug or drug.lower() == 'nan':
                continue
                
            condition = str(row['condition']) if not pd.isna(row['condition']) else "Not specified"
            review = str(row['review']) if not pd.isna(row['review']) else "No review available"
            rating = str(row['rating']) if not pd.isna(row['rating']) else "No rating"
            
            text = f"Drug: {drug}\nCondition: {condition}\nRating: {rating}\nReview: {review}"
            documents.append(Document(page_content=text, metadata={"drug": drug}))
            seen_drugs.add(drug)
        
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        
        st.success(f"Loaded {len(chunks)} document chunks for {len(seen_drugs)} unique drugs")
        return chunks, seen_drugs
        
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

chunks, seen_drugs = load_data()

# System prompt template
system_prompt = """You are a knowledgeable medical assistant specializing in medications. 
For any drug query, provide:
1. Primary uses and conditions treated
2. Effectiveness based on patient reviews
3. Common side effects
4. Important safety warnings

If the drug isn't in our database, state this clearly.
Never provide dosage recommendations - always advise consulting a doctor."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Semantic search function
def get_relevant_chunks(query, k=5):
    try:
        if not chunks:
            return []
            
        query_embed = embedder.encode(query, convert_to_tensor=True)
        chunk_texts = [c.page_content for c in chunks if isinstance(c, Document)]
        chunk_embeds = embedder.encode(chunk_texts, convert_to_tensor=True)
        
        scores = util.pytorch_cos_sim(query_embed, chunk_embeds)[0]
        top_k_idx = scores.topk(k).indices
        return [chunks[i] for i in top_k_idx]
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

# Query processing pipeline
def process_query(user_query):
    try:
        relevant_chunks = get_relevant_chunks(user_query)
        if not relevant_chunks:
            return "No information found about this medication in our database."
        
        # Create processing chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(question_answer_chain, memory)
        
        response = retrieval_chain.invoke({
            "input": user_query,
            "context": relevant_chunks
        })
        
        # Clean response
        cleaned_response = re.sub(r"<.*?>", "", response['answer']).strip()
        return cleaned_response
        
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return "Unable to process your query. Please try again."

# UI Components
st.title("💊 Medication Information System")
st.caption("Ask about any medication's uses, side effects, and patient reviews")

# Connection test
try:
    test_response = llm.invoke("What is 1+1?")
    st.sidebar.success("✅ AI model connected successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Connection failed: {str(e)}")
    st.stop()

# Display drug count
st.sidebar.markdown(f"**Database contains {len(seen_drugs)} medications**")
if st.sidebar.button("Show sample drugs"):
    st.sidebar.write(list(sorted(seen_drugs))[:15])

# Clear conversation button
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    memory.clear()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask about a medication..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.spinner("Researching medication..."):
        response = process_query(user_input)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
