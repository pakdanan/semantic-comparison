"""Streamlit app to compute similarity between sentences or paragraphs."""
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)
st.text(f'Sentence similarity: {embeddings}')
