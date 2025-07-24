import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd

# File paths
index_path = 'faiss_index.bin'
embeddings_path = 'embeddings.npy'
chunks_path = 'chunks.txt'
csv_path = 'Training Dataset.csv'

@st.cache_resource
def load_resources():
    chunks = load_chunks(chunks_path)
    index = faiss.read_index(index_path)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = None
    return chunks, index, embedder, df

# Load chunks
def load_chunks(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return [chunk.strip() for chunk in text.split('\n---\n') if chunk.strip()]

# Embed query
def embed_query(query, model):
    return model.encode([query])[0]

# Retrieve top k chunks
def retrieve(query, model, index, chunks, k=5):
    query_vec = embed_query(query, model)
    D, I = index.search(np.array([query_vec]), k)
    return [chunks[i] for i in I[0]]

# Generate answer using LLM
@st.cache_resource
def get_generator():
    return pipeline('text2text-generation', model='google/flan-t5-base')

def generate_answer(question, context):
    generator = get_generator()
    prompt = (
        "You are a helpful assistant for loan approval data. "
        "Use the context below to answer the user's question in a clear, concise way. "
        "If the answer is not in the context, say 'I don't know based on the data.'\n"
        f"Context:\n{context}\nQuestion: {question}"
    )
    result = generator(prompt, max_length=128, do_sample=False)
    return result[0]['generated_text']

# Streamlit UI
st.set_page_config(page_title="Loan Q&A Chatbot", page_icon="💬")
st.title("Loan Approval Q&A Chatbot")
st.write("Ask questions about the loan approval dataset!")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

chunks, index, embedder, df = load_resources()

question = st.text_input("Type your question:", "")
if st.button("Ask") and question:
    # Stats logic
    q_lower = question.lower()
    if df is not None:
        if "average applicant income" in q_lower:
            avg = df["ApplicantIncome"].mean()
            answer = f"The average applicant income is {avg:.2f}"
        elif "average coapplicant income" in q_lower:
            avg = df["CoapplicantIncome"].mean()
            answer = f"The average coapplicant income is {avg:.2f}"
        elif "average loan amount" in q_lower:
            avg = df["LoanAmount"].mean()
            answer = f"The average loan amount is {avg:.2f}"
        elif "average loan term" in q_lower or "average loan term in months" in q_lower:
            avg = df["Loan_Amount_Term"].mean()
            answer = f"The average loan term is {avg:.2f} months"
        elif "most frequent property area" in q_lower:
            area = df["Property_Area"].mode()[0]
            answer = f"The most frequent property area for loans is {area}"
        elif "approval rate" in q_lower or "loan approval rate" in q_lower:
            rate = (df["Loan_Status"] == "Y").mean() * 100
            answer = f"The loan approval rate is {rate:.2f}%"
        else:
            top_chunks = retrieve(question, embedder, index, chunks, k=5)
            context = '\n---\n'.join(top_chunks)
            answer = generate_answer(question, context)
    else:
        top_chunks = retrieve(question, embedder, index, chunks, k=5)
        context = '\n---\n'.join(top_chunks)
        answer = generate_answer(question, context)
    st.session_state.chat_history.append((question, answer))

if st.session_state.chat_history:
    st.subheader("Chat History")
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
