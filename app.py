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
import torch
import re
import os
import pandas as pd

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset and prepare document chunks
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

# Load language models for translation (optional, not used in main logic yet)
@st.cache_resource
def load_translation_models():
    ar_en_tokenizer = AutoTokenizer.from_pretrained("Abdalrahmankamel/NLLB_Egyptian_Arabic_to_English")
    ar_en_model = AutoModelForSeq2SeqLM.from_pretrained("Abdalrahmankamel/NLLB_Egyptian_Arabic_to_English")
    
    en_ar_tokenizer = AutoTokenizer.from_pretrained("NAMAA-Space/masrawy-english-to-egyptian-arabic-translator-v2.9")
    en_ar_model = AutoModelForSeq2SeqLM.from_pretrained("NAMAA-Space/masrawy-english-to-egyptian-arabic-translator-v2.9")
    
    return {
        'ar_en_tokenizer': ar_en_tokenizer,
        'ar_en_model': ar_en_model,
        'en_ar_tokenizer': en_ar_tokenizer,
        'en_ar_model': en_ar_model
    }

# Load main LLM model
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("aubmindlab/aragpt2-base", padding_side="left", truncation_side="left")
        if not hasattr(tokenizer, 'model_max_length') or tokenizer.model_max_length > 100000:
            tokenizer.model_max_length = 1024
        
        model = AutoModelForCausalLM.from_pretrained("aubmindlab/aragpt2-base", torch_dtype=torch.float32, low_cpu_mem_usage=True)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

with st.spinner("Loading model... This may take a few minutes the first time."):
    llm = load_model()

# Chat memory and prompt template
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
7. If the question is in Arabic, answer with Arabic only.

Return your answer as:
- **Patient Query:** Rephrased version of the user’s question
- **Doctor's Recommendations:** Suggested treatments based on the above
- **Safety Warnings (if any):** Explicit cautions based on the reviews or symptoms
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Patient: {input}\n\nMedical Records:\n{context}")
])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="input")

# Helper functions
def get_relevant_chunks(question, k=5):
    question_embedding = embedder.encode(question, convert_to_tensor=True)

    chunk_texts = []
    valid_chunks = []

    for c in chunks:
        text = getattr(c, "page_content", str(c))
        chunk_texts.append(text)
        valid_chunks.append(c)

    chunk_embeddings = embedder.encode(chunk_texts, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]
    top_k_idx = scores.argsort(descending=True)[:k]
    return [valid_chunks[i] for i in top_k_idx]

def clean_response(response_text):
    return re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

def detect_drug(query):
    query_lower = query.lower()
    for drug in seen_drugs:
        if drug.lower() in query_lower:
            return drug
    return None
def ask_question_with_memory(question, k=2):
    try:
        if not llm:
            return "System error: Model not loaded"

        question = str(question)[:300]

        relevant_chunks = get_relevant_chunks(question, k)
        context_text = "\n\n".join(
            getattr(doc, "page_content", str(doc)) for doc in relevant_chunks[:1]
        )

        memory.chat_memory.add_user_message(question[:200])

        chain = create_stuff_documents_chain(llm, prompt)
        result = chain.invoke({
            "input": question,
            "context": context_text,
            "chat_history": [msg for msg in memory.chat_memory.messages[-2:] if len(str(msg)) < 300]
        })

        response = clean_response(str(result))[:800]
        memory.chat_memory.add_ai_message(response)
        return response

    except Exception as e:
        return f"System error: {str(e)[:150]}"


# Streamlit App Interface
st.title("🤖 Medical Assistant Chatbot")
st.write("Ask me about medications, symptoms, or medical advice.")

# Initialize chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input and response
if user_input := st.chat_input("What is your medical question?"):
    if not user_input.strip():
        st.warning("Please enter a valid question.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Analyzing..."):
            try:
                response = ask_question_with_memory(user_input)
                if "error" in response.lower():
                    st.error("Response error")
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.write(response)
            except Exception as e:
                st.error(f"System error: {str(e)[:200]}")
