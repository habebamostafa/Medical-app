import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from transformers import GenerationConfig

import torch
import re
import os
import pandas as pd
# تهيئة نموذج التضمين
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# تحميل مجموعة البيانات
@st.cache_resource
def load_data():
    train_data = pd.read_csv("data.csv")
    
    seen_drugs = set()
    documents = []
    
    for _, example in train_data.iterrows():
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

chunks, seen_drugs = load_data()

# تهيئة نموذج الترجمة
@st.cache_resource
def load_translation_models():
    # Arabic to English
    ar_en_tokenizer = AutoTokenizer.from_pretrained("Abdalrahmankamel/NLLB_Egyptian_Arabic_to_English")
    ar_en_model = AutoModelForSeq2SeqLM.from_pretrained("Abdalrahmankamel/NLLB_Egyptian_Arabic_to_English")
    
    # English to Arabic
    en_ar_tokenizer = AutoTokenizer.from_pretrained("NAMAA-Space/masrawy-english-to-egyptian-arabic-translator-v2.9")
    en_ar_model = AutoModelForSeq2SeqLM.from_pretrained("NAMAA-Space/masrawy-english-to-egyptian-arabic-translator-v2.9")
    
    return {
        'ar_en_tokenizer': ar_en_tokenizer,
        'ar_en_model': ar_en_model,
        'en_ar_tokenizer': en_ar_tokenizer,
        'en_ar_model': en_ar_model
    }


@st.cache_resource
def load_model():
    model_name = "aubmindlab/aragpt2-base"
    
    try:
        # Load tokenizer with proper settings
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            truncation_side="left",
            model_max_length=1024
        )
        
        # Load model with safe defaults
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # Create pipeline with safe defaults
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,  # Conservative output length
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            device=0 if torch.cuda.is_available() else -1
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

with st.spinner("جاري تحميل النموذج... قد يستغرق عدة دقائق لأول مرة"):
    llm = load_model()
# إعداد الذاكرة والموجه
prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You are a highly qualified and empathetic medical doctor specializing in pharmacology and diagnostics.
Your task is to assist patients by analyzing symptoms and suggesting possible over-the-counter or prescription treatments,
using only the provided medical context and patient history.

Please follow these guidelines strictly:
1. Summarize or rephrase patient reviews professionally — focus on effectiveness, side effects, and usage patterns.
2. Only recommend medications based on the drug context. Do NOT invent drugs or diagnoses not found in the context.
3. If reviews indicate serious side effects, mention them clearly as warnings.
4. If symptoms are severe, radiating, or persistent, recommend immediate consultation with a licensed healthcare provider.
5. If the patient asks about **dosage, price, amount, or frequency**, respond clearly that they must refer to a licensed doctor or read the official instructions — do not make assumptions or guesses.
6. Write in a professional tone, like a medical consultation summary. Avoid overly casual language.
7.if the question is in arabic, answer with arabic only

Return your answer as:
- **Patient Query:** Rephrased version of the user’s question
- **Doctor's Recommendations:** Suggested treatments based on the above
- **Safety Warnings (if any):** Explicit cautions based on the reviews or symptoms
"""),

    MessagesPlaceholder(variable_name="chat_history"),

    ("human", "Patient: {input}\n\nMedical Records:\n{context}")
])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,input_key="input")

# وظائف المساعدة
def get_relevant_chunks(question, k=5):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    chunk_embeddings = embedder.encode([c.page_content for c in chunks], convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]
    top_k_idx = scores.argsort(descending=True)[:k]
    return [chunks[i] for i in top_k_idx]

def clean_response(response_text):
    return re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

def detect_drug(query):
    query_lower = query.lower()
    for drug in seen_drugs:
        if drug.lower() in query_lower:
            return drug
    return None

def ask_question_with_memory(question, k=2):  # Reduced chunks to be safer
    try:
        if not llm:
            return "Model not loaded properly. Please refresh the page."
        
        # Simple input validation
        question = str(question)[:500]  # Hard truncate to prevent errors
        
        # Get relevant chunks with safe fallback
        try:
            relevant_chunks = get_relevant_chunks(question, k)
            context = "\n\n".join([chunk.page_content[:500] for chunk in relevant_chunks][:2])
        except:
            context = ""
        
        # Add to memory (with size limit)
        memory.chat_memory.add_user_message(question[:300])
        
        # Safely create chain
        try:
            chain = create_stuff_documents_chain(llm, prompt)
            result = chain.invoke({
                "input": question,
                "context": relevant_chunks if len(relevant_chunks) > 0 else [Document(page_content="No context found")],
                "chat_history": memory.chat_memory.messages[-2:]  # Only last 2 messages
            })
            return clean_response(result)[:1000]  # Limit response length
        except Exception as e:
            return f"I couldn't generate a proper response. Error: {str(e)[:200]}"
            
    except Exception as e:
        return "Sorry, I encountered an unexpected error. Please try again."
    

st.title("🤖 Medical Assistant Chatbot")
st.write("Ask me about medications, symptoms, or medical advice.")

# تهيئة حالة المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض رسائل المحادثة السابقة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize model
llm = load_model()
if llm is None:
    st.error("Failed to initialize the AI model. Please check your setup.")
    st.stop()  # Prevent further execution

# ... (rest of your setup code remains the same) ...

if user_input := st.chat_input("What is your medical question?"):
    # Simple input validation
    if not user_input.strip():
        st.warning("Please enter a question")
        st.stop()
    
    with st.chat_message("user"):
        st.write(user_input)
    
    try:
        with st.spinner("Thinking..."):
            response = ask_question_with_memory(user_input)
            
        with st.chat_message("assistant"):
            st.write(response if response else "No response generated")
            
    except Exception as e:
        st.error(f"Application error: {str(e)[:200]}")
        st.write("Sorry, I couldn't process that request. Please try again.")