import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import pdfplumber
import pandas as pd

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(text):
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
    embedding = mean_pooling(output, inputs['attention_mask'])
    return embedding.squeeze(0).numpy()

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

def extract_text_from_pdf_uploaded(uploaded_file):
    if uploaded_file is not None:
    # Read the PDF directly from memory
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = "\n".join([page.get_text("text") for page in doc])
    return text

def extract_text_and_tables(uploaded_file):
    """Extracts both text and tables from a PDF file."""
    print('start extract text and table')
    text = ""
    tables = []

    # Extract text using PyMuPDF
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    print('finish extract text')
    # Extract tables using pdfplumber
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:  # Skip empty tables
                df = pd.DataFrame(table)  # Convert to DataFrame
                tables.append(df)
    print('finish extract table')
    return text, tables 

def table_to_text(table):
    """Converts a DataFrame table into a structured text format for embedding."""
    return "\n".join([" | ".join(map(str, row)) for row in table.values])

def chunk_to_vector(pdf_text, embed_table, embed_model, chunk_size=2000, chunk_overlap=500):
    # Set up the text splitter with proper numeric chunk size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # print('text splitted')
    # print(embed_table)
    chunks = text_splitter.split_text(pdf_text)
    # print('chunk added')
    if not chunks:
        print("Error: No chunks created!")
        return None
    # for chunk in chunks:
        # print(chunk)
    all_chunks = chunks + embed_table
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_db = FAISS.from_texts(all_chunks, embedding_model)

    return vector_db

def set_llm(model, upload_file):
    try:
        pdf_text, pdf_table = extract_text_and_tables(upload_file)
        embed_table = [table_to_text(table) for table in pdf_table]
        vector_db = chunk_to_vector(pdf_text, embed_table, "emilyalsentzer/Bio_ClinicalBERT")
        llm = Ollama(model=model)
        retriever = vector_db.as_retriever()  # Ensure retriever is correct
        rag_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
        query = """
        Please extract the following information from the Case Report:
        1. **Basic Patient Information** (Name, Gender, Age, Chief Complaint)
        2. **Disease Progression** (Onset Time, Symptom Changes)
        3. **Treatment** (Surgery, Medication, Other Therapies)
        4. **Key Findings** (Imaging, Pathology, Genetic Test Results)
        5. **Post-Operative Status** (Recovery Progress, Complications, Follow-up)
        """
        response = rag_chain.run(query)
        
        return response, rag_chain
    except Exception as e:
        return f"RAG 執行錯誤：{str(e)}"

st.title("Case Report Analyze System")
USER_CREDENTIALS = {"admin": "stmedical"}

def login():
    """Simple login system for private access."""
    st.title("🔒 Secure Login")

    with st.form(key="login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")  # Works with "Enter" or button

    if submit_button:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["logged_in"] = True
            st.success("✅ Login successful!")
            st.rerun() 
        else:
            st.error("❌ Incorrect username or password!")

# Check if the user is logged in
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
else:
    uploaded_file = st.file_uploader("Please upload a Case Report (PDF)", type=["pdf"])

    if uploaded_file is not None:
        st.write("Analyzing...")
        summary, rag_chain = set_llm('llama3.2', uploaded_file)
        st.subheader("Case Summarize")
        st.write(summary)
        if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

        user_question = st.text_input("🔎 Ask any question about the case report:")

        if st.button("Get Answer") and user_question:
            response = rag_chain.invoke(user_question)  # ✅ Retrieve answer using LLM
            st.session_state.chat_history.append(("You", user_question))
            st.session_state.chat_history.append(("AI", response))
            st.write(response)