import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd

index_path = 'faiss_index.bin'
embeddings_path = 'embeddings.npy'
chunks_path = 'chunks.txt'
csv_path = 'Training Dataset.csv'


def load_chunks(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return [chunk.strip() for chunk in text.split('\n---\n') if chunk.strip()]


def load_index_and_embeddings():
    index = faiss.read_index(index_path)
    embeddings = np.load(embeddings_path)
    return index, embeddings


def embed_query(query, model):
    return model.encode([query])[0]


def retrieve(query, model, index, chunks, k=7):
    query_vec = embed_query(query, model)
    D, I = index.search(np.array([query_vec]), k)
    return [chunks[i] for i in I[0]]


def generate_answer(question, context):
    generator = pipeline('text2text-generation', model='google/flan-t5-large')
    prompt = (
        "You are a helpful assistant for loan approval data. "
        "Use the context below to answer the user's question in a clear, complete sentence. "
        "If the answer is not in the context, say 'I don't know based on the data.'\n"
        f"Context:\n{context}\nQuestion: {question}"
    )
    result = generator(prompt, max_length=256, do_sample=False)
    return result[0]['generated_text']

if __name__ == "__main__":
    print("Loading resources...")
    chunks = load_chunks(chunks_path)
    index, embeddings = load_index_and_embeddings()
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    # Load CSV for stats
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Warning: Could not load CSV for stats: {e}")
        df = None

    print("Ready! Type your question below.")
    while True:
        question = input("You: ")
        if question.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        
        q_lower = question.lower()
        if df is not None:
            if "average applicant income" in q_lower:
                avg = df["ApplicantIncome"].mean()
                print(f"Bot: The average applicant income is {avg:.2f}")
                continue
            elif "average coapplicant income" in q_lower:
                avg = df["CoapplicantIncome"].mean()
                print(f"Bot: The average coapplicant income is {avg:.2f}")
                continue
            elif "average loan amount" in q_lower:
                avg = df["LoanAmount"].mean()
                print(f"Bot: The average loan amount is {avg:.2f}")
                continue
            elif "average loan term" in q_lower or "average loan term in months" in q_lower:
                avg = df["Loan_Amount_Term"].mean()
                print(f"Bot: The average loan term is {avg:.2f} months")
                continue
            elif "most frequent property area" in q_lower:
                area = df["Property_Area"].mode()[0]
                print(f"Bot: The most frequent property area for loans is {area}")
                continue
            elif "approval rate" in q_lower or "loan approval rate" in q_lower:
                rate = (df["Loan_Status"] == "Y").mean() * 100
                print(f"Bot: The loan approval rate is {rate:.2f}%")
                continue
            elif ("married applicants" in q_lower and ("approved" in q_lower or "approval" in q_lower or "get their loan" in q_lower)) or ("are married applicants more likely" in q_lower):
                married = df[df["Married"] == "Yes"]
                unmarried = df[df["Married"] == "No"]
                married_rate = (married["Loan_Status"] == "Y").mean() * 100
                unmarried_rate = (unmarried["Loan_Status"] == "Y").mean() * 100
                print(f"Bot: Approval rate for married applicants is {married_rate:.2f}%. For unmarried applicants, it is {unmarried_rate:.2f}%.")
                continue

        top_chunks = retrieve(question, embedder, index, chunks, k=7)
        context = '\n---\n'.join(top_chunks)
        answer = generate_answer(question, context)
        print(f"Bot: {answer}\n")
