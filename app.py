import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import google.generativeai as genai
import os

# --- Gemini API Key Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))
if not GEMINI_API_KEY:
    st.error("Please set your Gemini API key as an environment variable or Streamlit secret.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)

# --- Gemini LLM Helper Function ---
def ask_gemini(prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error from Gemini: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="PDF Chatbot (Gemini)", layout="wide")
st.title("PDF Chatbot (Gemini RAG)")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    # --- Extract PDF Text & Chunk ---
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text

    def chunk_text(text, chunk_size=500, chunk_overlap=50):
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i+chunk_size]
            chunks.append(' '.join(chunk))
            i += chunk_size - chunk_overlap
        return chunks
    chunks = chunk_text(raw_text)

    # --- Embedding ---
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    # --- FAISS Vector DB ---
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # --- Question UI ---
    user_query = st.text_input("Ask a question about your PDF:")

    if user_query:
        # --- Similarity Search ---
        query_embedding = model.encode([user_query])
        D, I = index.search(np.array(query_embedding), k=3)
        relevant_chunks = [chunks[i] for i in I[0]]

        # --- Prompt Construction ---
        context = "\n\n".join(relevant_chunks)
        prompt = f"""
You are an assistant. Answer the question using only the context below.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{user_query}
"""

        # --- Gemini LLM Call ---
        answer = ask_gemini(prompt)
        st.write("**Answer:**", answer)

        # --- Chat History ---
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append({"question": user_query, "answer": answer})
        st.write("---")
        st.write("**Chat History:**")
        for qa in st.session_state.chat_history:
            st.write(f"Q: {qa['question']}")
            st.write(f"A: {qa['answer']}")
else:
    st.info("Please upload a PDF to get started.")

# --- Sidebar: Deployment Instructions ---
st.sidebar.header("Deployment Instructions")
st.sidebar.markdown("""
**requirements.txt** example:

