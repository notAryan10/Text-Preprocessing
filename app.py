import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

st.title("Text Preprocessing → Embeddings Visualizer")

text = st.text_area("Enter your text here:")

if text:
    st.write("You entered:")
    st.write(text)

st.write("---")


st.sidebar.header("Preprocessing Controls")
lowercase = st.sidebar.checkbox("Lowercase", True)
remove_punct = st.sidebar.checkbox("Remove Punctuation", True)
remove_stopwords = st.sidebar.checkbox("Remove Stopwords", True)

tokenizer_type = st.sidebar.selectbox(
    "Tokenization Type",
    ["Word", "Character"]
)

vectorizer_type = st.sidebar.selectbox(
    "Vectorization Method",
    ["Bag of Words", "TF-IDF"]
)


def clean_text(text, lowercase=True, remove_punct=True):
    original = text

    if lowercase:
        text = text.lower()

    if remove_punct:
        text = re.sub(r'[^\w\s]', '', text)

    return original, text


def tokenize(text, method="Word"):
    if method == "Word":
        return text.split()
    elif method == "Character":
        return list(text)


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def normalize_tokens(tokens, remove_stopwords=True):
    processed = []
    for word in tokens:
        if remove_stopwords and word in stop_words:
            continue
        lemma = lemmatizer.lemmatize(word)
        processed.append(lemma)
    return processed



def build_vocab(tokens):
    vocab = Counter(tokens)
    word_to_index = {word: i for i, word in enumerate(vocab.keys())}
    return vocab, word_to_index

def vectorize(text, method="Bag of Words"):
    if method == "Bag of Words":
        vec = CountVectorizer()
    else:
        vec = TfidfVectorizer()

    X = vec.fit_transform([text])
    return vec.get_feature_names_out(), X.toarray()



if text:
    original, cleaned = clean_text(text, lowercase, remove_punct)

    st.subheader("Text Cleaning")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Original Text")
        st.write(original)

    with col2:
        st.write("Cleaned Text")
        st.write(cleaned)
    tokens = tokenize(cleaned, tokenizer_type)

    st.subheader("Tokenization")
    st.write(tokens)

    normalized_tokens = normalize_tokens(tokens, remove_stopwords)

    st.subheader("Normalization (Stopwords Removal + Lemmatization)")
    st.write(normalized_tokens)

    vocab, word_to_index = build_vocab(normalized_tokens)

    st.subheader("Vocabulary")
    vocab_df = pd.DataFrame({
        "Word": list(word_to_index.keys()),
        "Index": list(word_to_index.values()),
        "Frequency": list(vocab.values())
    })

    st.dataframe(vocab_df)

    st.subheader("Vectorization")

    words, vectors = vectorize(" ".join(normalized_tokens), vectorizer_type)

    vec_df = pd.DataFrame(vectors, columns=words)
    st.dataframe(vec_df)


    st.subheader("Word Similarity")
    
    @st.cache_resource
    def load_model():
        return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    model = load_model()

    input1 = st.text_input("Word 1")
    input2 = st.text_input("Word 2")

    word1 = input1.strip().lower()
    word2 = input2.strip().lower()

    if word1 and word2:
        if word1 in normalized_tokens and word2 in normalized_tokens:
            emb = model.encode([word1, word2])
            sim = cosine_similarity(emb[0:1], emb[1:2])[0][0]
            st.metric("Similarity Score", f"{sim:.4f}")
        else:
            missing = []
            if word1 not in normalized_tokens: missing.append(f"'{input1}'")
            if word2 not in normalized_tokens: missing.append(f"'{input2}'")
            st.warning(f"Word(s) {', '.join(missing)} not found in the current text tokens. Please use words that appear in the 'Normalization' section above.")