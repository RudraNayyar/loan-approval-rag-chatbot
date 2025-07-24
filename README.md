# Loan Approval RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for answering questions about loan approval using document retrieval and generative AI.

## Features
- Document retrieval with FAISS and sentence-transformers
- Generative answers using Hugging Face LLMs
- Statistical and analytical answers using pandas
- Streamlit web UI

## How to Run

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Prepare data and build index:**
   ```
   python prepare_data.py
   python build_index.py
   ```

3. **Run chatbot:**
   ```
   python qa_bot.py
   ```

4. **Run Streamlit UI:**
   ```
   streamlit run app.py
   ```

## Dataset

Uses [Kaggle Loan Approval Dataset](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction).

## License

MIT
