import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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


# تهيئة نموذج المحادثة
@st.cache_resource
def init_llm():
    return ChatOllama(
        model="deepseek-r1:1.5b",
        base_url=os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    )

llm = init_llm()

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
    ("human", "{input}")
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

def ask_question_with_memory(question, k=5):
    try:
        # الحصول على الأجزاء ذات الصلة
        relevant_chunks = get_relevant_chunks(question, k)
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        
        # إضافة رسالة المستخدم إلى الذاكرة
        memory.chat_memory.add_user_message(question)
        
        # إنشاء السلسلة
        chain = create_stuff_documents_chain(llm, prompt)
        
        # استدعاء السلسلة مع المدخلات الصحيحة
        result = chain.invoke({
            "input": input,  # المفتاح يجب أن يكون "input"
            "context": context,
            "chat_history": memory.chat_memory.messages
        })
        
        cleaned = clean_response(result)
        memory.chat_memory.add_ai_message(cleaned)
        return cleaned
        
    except Exception as e:
        st.error(f" error: {str(e)}")
        return "try agin"

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

# معالجة الإدخال الجديد
if prompt := st.chat_input("What is your medical question?"):
    # إضافة رسالة المستخدم إلى حالة المحادثة
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # الحصول على الرد
    with st.spinner("Thinking..."):
        response = ask_question_with_memory(prompt)
    
    # عرض الرد
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # إضافة رد المساعد إلى حالة المحادثة
    st.session_state.messages.append({"role": "assistant", "content": response})