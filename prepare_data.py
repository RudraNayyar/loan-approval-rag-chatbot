import pandas as pd

# Load the loan approval dataset
DATA_PATH = 'Training Dataset.csv'

def load_data(path=DATA_PATH):
    """
    Loads the CSV file and returns a pandas DataFrame.
    """
    try:
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} rows from {path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def chunk_data(df):
    """
    Converts each row into a text chunk for retrieval.
    Returns a list of strings, each representing a document.
    """
    chunks = []
    for idx, row in df.iterrows():
        # Combine all columns into a readable string
        chunk = '\n'.join([f"{col}: {row[col]}" for col in df.columns])
        chunks.append(chunk)
    print(f"Created {len(chunks)} text chunks.")
    return chunks

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        chunks = chunk_data(df)
        # Save chunks for later use
        with open('chunks.txt', 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(chunk + '\n---\n')
        print("Chunks saved to chunks.txt")
