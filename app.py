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
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 2048  # Set maximum token length
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        truncation=True,
        max_length=1024  # Explicit max length for input
    )
    
    return HuggingFacePipeline(pipeline=pipe)

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

def ask_question_with_memory(question, k=3):
    try:
        # Tokenize and truncate the question
        tokenizer = llm.pipeline.tokenizer
        inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
        truncated_question = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        
        # Get relevant chunks
        relevant_chunks = get_relevant_chunks(truncated_question, k)
        
        # Prepare safe context
        context_parts = []
        remaining_length = 800  # Target context length in tokens
        
        for chunk in relevant_chunks:
            chunk_text = chunk.page_content
            chunk_tokens = tokenizer(chunk_text, return_tensors="pt")["input_ids"]
            
            if len(chunk_tokens[0]) <= remaining_length:
                context_parts.append(chunk_text)
                remaining_length -= len(chunk_tokens[0])
            else:
                truncated_chunk = tokenizer.decode(chunk_tokens[0][:remaining_length], skip_special_tokens=True)
                context_parts.append(truncated_chunk + "... [truncated]")
                break
        
        context = "\n\n".join(context_parts)
        
        # Process with memory
        memory.chat_memory.add_user_message(truncated_question)
        
        chain = create_stuff_documents_chain(llm, prompt)
        result = chain.invoke({
            "input": truncated_question,
            "context": relevant_chunks,
            "chat_history": memory.chat_memory.messages[-4:]  # Last 4 messages
        })
        
        cleaned = clean_response(result)
        memory.chat_memory.add_ai_message(cleaned)
        return cleaned
        
    except Exception as e:
        st.error(f"Detailed error: {str(e)}")
        return "I'm having trouble answering that. Could you please rephrase your question?"
# واجهة Streamlit
st.title("🤖 Medical Assistant Chatbot")
st.write("Ask me about medications, symptoms, or medical advice.")

# تهيئة حالة المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض رسائل المحادثة السابقة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("What is your medical question?"):
    if not user_input.strip():
        st.warning("Please enter a valid question")
    else:
        # Process the question
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.spinner("Analyzing your question..."):
            try:
                response = ask_question_with_memory(user_input)
            except Exception as e:
                response = "I encountered a technical issue. Please try a different question."
                st.error(f"System error: {str(e)}")
        
        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})