import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    llm = ChatOllama(model="deepseek-r1:1.5b", base_url="https://stupid-buses-draw.loca.lt")
    return embedder, llm

embedder, llm = load_models()

# Robust data loading with error handling
@st.cache_resource
def load_data():
    try:
        train_data = pd.read_csv("data.csv")
        seen_drugs = set()
        documents = []
        
        # Validate required columns exist
        required_columns = {'drugName', 'condition', 'review', 'rating'}
        if not required_columns.issubset(train_data.columns):
            missing = required_columns - set(train_data.columns)
            raise ValueError(f"Missing columns in data: {missing}")

        for _, row in train_data.iterrows():
            try:
                drug = str(row['drugName']).strip()
                if not drug or pd.isna(drug):
                    continue
                    
                if drug not in seen_drugs:
                    condition = str(row['condition']) if not pd.isna(row['condition']) else "Unknown condition"
                    review = str(row['review']) if not pd.isna(row['review']) else "No review available"
                    rating = str(row['rating']) if not pd.isna(row['rating']) else "No rating"
                    
                    text = f"Drug: {drug}\nCondition: {condition}\nRating: {rating}\nReview: {review}"
                    documents.append(Document(page_content=text))
                    seen_drugs.add(drug)
            except Exception as e:
                st.warning(f"Skipping row due to error: {str(e)}")
                continue

        # Split documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        
        return chunks, seen_drugs
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return empty but properly structured data
        return [], set()

chunks, seen_drugs = load_data()

# System prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You are a medical expert providing information about medications.
For any drug query, provide:
1. Common uses/conditions treated
2. Effectiveness based on reviews
3. Notable side effects
4. Safety warnings

Only use information from the provided context.
If unsure, say "Consult a healthcare professional"."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Question: {input}\nContext:\n{context}")
])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="input")

# Improved chunk retrieval
def get_relevant_chunks(query, k=5):
    try:
        if not chunks:
            st.warning("No medication data available")
            return []
            
        query_embed = embedder.encode(query, convert_to_tensor=True)
        chunk_texts = [c.page_content for c in chunks if hasattr(c, 'page_content')]
        
        if not chunk_texts:
            st.warning("No valid medication records found")
            return []
            
        chunk_embeds = embedder.encode(chunk_texts, convert_to_tensor=True)
        sim_scores = util.pytorch_cos_sim(query_embed, chunk_embeds)[0]
        top_indices = sim_scores.argsort(descending=True)[:k]
        return [chunks[i] for i in top_indices]
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def clean_response(text):
    return re.sub(r"<.*?>", "", str(text)).strip()

def ask_question(question):
    try:
        # Get relevant information
        relevant_chunks = get_relevant_chunks(question)
        if not relevant_chunks:
            return "No medication information found. Please try another query."
            
        context = "\n---\n".join([c.page_content for c in relevant_chunks])
        
        # Add to conversation history
        memory.chat_memory.add_user_message(question)
        
        # Generate response
        chain = create_stuff_documents_chain(llm, prompt)
        response = chain.invoke({
            "input": question,
            "context": context,
            "chat_history": memory.chat_memory.messages
        })
        
        # Clean and store response
        cleaned = clean_response(response)
        memory.chat_memory.add_ai_message(cleaned)
        return cleaned
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "Unable to process request. Please try again."

# Streamlit UI
st.title("💊 Medication Information Assistant")
st.write("Ask about any medication's uses, effects, and reviews")

# Display drug count
if seen_drugs:
    st.sidebar.markdown(f"**{len(seen_drugs)} medications available**")
    if st.sidebar.button("Show sample drugs"):
        st.sidebar.write(list(sorted(seen_drugs))[:20])

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about a medication..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.spinner("Researching..."):
        response = ask_question(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)