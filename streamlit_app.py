import streamlit as st
import joblib
import spacy
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Veritas AI: The Truth Detector",
    page_icon="🤖",
    layout="wide"
)

# --- Constants ---
MIN_CHARS = 250 # Define the minimum character count for analysis

# --- Caching the Models ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_random_forest_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        nlp = spacy.load("en_core_web_sm")
        print("✅ Models loaded successfully.")
        return model, vectorizer, nlp
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return None, None, None

model, vectorizer, nlp = load_models()

# --- Preprocessing Function ---
def preprocess_text(text):
    doc = nlp(text)
    processed_tokens = [
        token.lemma_.lower() 
        for token in doc 
        if not token.is_stop and not token.is_punct
    ]
    return " ".join(processed_tokens)

# --- Main App Interface ---
st.title("Veritas AI: The Truth Detector 🤖")
st.write(f"An AI detector powered by a Random Forest model with **99.77% accuracy**. For best results, please enter text with at least **{MIN_CHARS}** characters.")

# --- Sidebar ---
st.sidebar.header("About This App")
st.sidebar.write("""
This application uses a machine learning model to distinguish between human-written and AI-generated text.
**Workflow:**
1.  **Preprocessing:** The input text is cleaned using spaCy.
2.  **Vectorization:** The cleaned text is converted into numerical features using TF-IDF.
3.  **Prediction:** A pre-trained Random Forest model predicts the origin of the text.
""")
st.sidebar.warning(f"**Note:** This model was trained on long-form essays and is most accurate with texts longer than {MIN_CHARS} characters.")

# --- Input and Prediction ---
if model is None or vectorizer is None:
    st.error("Model files are not loaded. Please check the logs.")
else:
    # Initialize session state to hold the text
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""

    def update_text():
        st.session_state.input_text = st.session_state.widget_text

    input_text = st.text_area(
        "Enter the text you want to analyze:", 
        height=250, 
        placeholder="Paste your text here...",
        key="widget_text",
        on_change=update_text
    )

    # --- Character Counter ---
    char_count = len(st.session_state.input_text)
    if char_count < MIN_CHARS:
        st.warning(f"Characters: {char_count}/{MIN_CHARS}")
    else:
        st.success(f"Characters: {char_count}/{MIN_CHARS}")

    # --- Analyze Button ---
    # The button is disabled if the character count is too low
    if st.button("Analyze Text", disabled=(char_count < MIN_CHARS)):
        with st.spinner("Analyzing..."):
            processed_text = preprocess_text(st.session_state.input_text)
            vectorized_text = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_text)[0]
            probabilities = model.predict_proba(vectorized_text)[0]

            col1, col2 = st.columns(2)
            
            if prediction == 1:
                label = "AI-Generated"
                confidence = probabilities[1]
                col1.error(f"**Prediction: {label}**")
            else:
                label = "Human-Written"
                confidence = probabilities[0]
                col1.success(f"**Prediction: {label}**")
            
            col2.metric(label="Confidence", value=f"{confidence:.2%}")
            st.progress(confidence)