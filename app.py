import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub

# Initialize models
@st.cache_resource

def load_models():
    hf_token = st.secrets["huggingfacehub_api_token"]
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    llm = HuggingFaceHub(
        repo_id="deepseek-ai/deepseek-llm-7b-chat",
        model_kwargs={
            "temperature": 0.3,
            "max_new_tokens": 512
        },
        huggingfacehub_api_token=hf_token
    )
    return embedder, llm

embedder, llm = load_models()

# Data loading with robust error handling
@st.cache_resource
def load_data():
    try:
        train_data = pd.read_csv("data.csv")
        seen_drugs = set()
        documents = []
        
        required_columns = {'drugName', 'condition', 'review', 'rating'}
        if not required_columns.issubset(train_data.columns):
            missing = required_columns - set(train_data.columns)
            st.error(f"Missing required columns: {missing}")
            return [], set()

        for _, row in train_data.iterrows():
            try:
                drug = str(row['drugName']).strip()
                if not drug or drug.lower() == 'nan':
                    continue
                    
                condition = str(row['condition']) if not pd.isna(row['condition']) else "Not specified"
                review = str(row['review']) if not pd.isna(row['review']) else "No review available"
                rating = str(row['rating']) if not pd.isna(row['rating']) else "No rating"
                
                text = f"Drug: {drug}\nCondition: {condition}\nRating: {rating}\nReview: {review}"
                documents.append(Document(page_content=text, metadata={"drug": drug}))
                seen_drugs.add(drug)
            except Exception as e:
                st.warning(f"Skipping row due to error: {str(e)}")
                continue

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        
        st.success(f"Loaded {len(chunks)} chunks for {len(seen_drugs)} unique drugs")
        return chunks, seen_drugs
        
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return [], set()

chunks, seen_drugs = load_data()

# System prompt template
system_prompt = """You are a knowledgeable medical assistant specializing in medications. 
For any drug query, provide accurate information including:
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

# Initialize memory and chains properly
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def get_relevant_chunks(query, k=5):
    try:
        if not chunks:
            return []
            
        query_embed = embedder.encode(query, convert_to_tensor=True)
        valid_chunks = [c for c in chunks if isinstance(c, Document) and hasattr(c, 'page_content')]
        
        if not valid_chunks:
            return []
            
        chunk_texts = [c.page_content for c in valid_chunks]
        chunk_embeds = embedder.encode(chunk_texts, convert_to_tensor=True)
        
        # Ensure compatible dimensions
        if query_embed.shape[0] != chunk_embeds.shape[0]:
            query_embed = query_embed.unsqueeze(0)
            
        scores = util.pytorch_cos_sim(query_embed, chunk_embeds)[0]
        top_k_idx = scores.argsort(descending=True)[:k]
        return [valid_chunks[i] for i in top_k_idx]
        
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return []

def process_query(user_query):
    try:
        # Retrieve relevant context
        relevant_chunks = get_relevant_chunks(user_query)
        if not relevant_chunks:
            return "No information found about this medication in our database."
        
        # Create the processing chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        
        # Combine with memory
        retrieval_chain = create_retrieval_chain(
            question_answer_chain,
            memory
        )
        
        # Prepare context documents
        context_docs = [Document(page_content=chunk.page_content) for chunk in relevant_chunks]
        
        # Invoke the chain
        response = retrieval_chain.invoke({
            "input": user_query,
            "context": context_docs
        })
        
        # Clean and return response
        return re.sub(r"<.*?>", "", response['answer']).strip()
        
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return "Unable to process your query. Please try again."

# Streamlit UI
st.title("💊 Medication Information System")
st.caption("Ask about any medication's uses, side effects, and patient reviews")
try:
    test = llm.invoke("What is Bupropion?")
    st.sidebar.success("✅ Ollama is connected successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Ollama connection failed: {e}")

# Display drug count
if seen_drugs:
    st.sidebar.markdown(f"**Database contains {len(seen_drugs)} medications**")
    if st.sidebar.button("Show sample drugs"):
        st.sidebar.write(list(sorted(seen_drugs))[:15])

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
