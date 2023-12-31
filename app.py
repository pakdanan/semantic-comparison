import streamlit as st
from sentence_transformers import SentenceTransformer, util

def compute_similarity(sentence1: str, sentence2: str, model) -> str:
    query_embeddings = model.encode(sentence1)
    passage_embedding = model.encode(sentence2)
    scores = util.cos_sim(query_embeddings, passage_embedding)
    return str(scores.numpy()[0][0])

# Render Streamlit page
st.header("Compute Similarity between sentences or paragraphs.")
st.markdown(
    "This mini-app computes similarity between two or multiple sentences using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model."
)

@st.cache_resource  # 👈 Add the caching decorator
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
model = load_model()

sentence1 = st.text_input(label="Sentence 1", placeholder="Text goes here...")
sentence2 = st.text_input(label="Sentence 2", placeholder="Text goes here...")

if st.button(
    label="Compute similarity",
    type="primary",
):
    result = compute_similarity(sentence1, sentence2, model)
    st.text(f'Sentence similarity: {result}')

text_spinner_placeholder = st.empty()
