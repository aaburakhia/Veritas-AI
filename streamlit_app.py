import streamlit as st
import joblib
import spacy
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Veritas AI | The Truth Detector",
    page_icon="🔎",
    layout="wide"
)

# --- Constants ---
MIN_CHARS = 250 # Define the minimum character count for analysis

# --- Caching the Models ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load('model/best_random_forest_model.joblib')
        vectorizer = joblib.load('model/tfidf_vectorizer.joblib')
        nlp = spacy.load("en_core_web_sm")
        print("✅ Models loaded successfully.")
        return model, vectorizer, nlp
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        st.error(f"Error loading model files: {e}")
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

# --- Sidebar ---
st.sidebar.header("About Veritas AI 🔎")
st.sidebar.write("""
This application uses a machine learning model to distinguish between human-written and AI-generated text.

It is powered by a **Random Forest** model with **99.77% accuracy** on its test dataset.
""")
st.sidebar.info(f"**Note:** The model is most accurate with texts longer than {MIN_CHARS} characters, as it was trained on long-form essays.")
st.sidebar.markdown("---")
st.sidebar.write("Project by: [Your Name Here]") # <-- Change this to your name!

# --- Main App Interface ---
st.title("Veritas AI: The Truth Detector")
st.write("Paste the text you want to analyze below. The model will determine its likely origin.")
st.markdown("---")

# --- Two-Column Layout ---
col1, col2 = st.columns([2, 1.5])

# --- Column 1: Input ---
with col1:
    st.subheader("Your Text")
    # THIS IS THE SIMPLIFIED AND CORRECTED INPUT WIDGET
    input_text = st.text_area("Enter text here:", height=350, placeholder="Start typing or paste your text...")
    char_count = len(input_text)
    
    # This counter will now update automatically on every interaction.
    if char_count < MIN_CHARS:
        st.warning(f"Characters: {char_count}/{MIN_CHARS}")
    else:
        st.success(f"Characters: {char_count}/{MIN_CHARS}")

# --- Column 2: Output ---
with col2:
    st.subheader("Analysis Results")
    
    # The button now correctly checks the length of the 'input_text' variable.
    if st.button("Analyze Text", disabled=(char_count < MIN_CHARS), use_container_width=True):
        if model is not None and vectorizer is not None:
            with st.spinner("Analyzing..."):
                processed_text = preprocess_text(input_text)
                vectorized_text = vectorizer.transform([processed_text])
                prediction = model.predict(vectorized_text)[0]
                probabilities = model.predict_proba(vectorized_text)[0]

                if prediction == 1:
                    label = "AI-Generated"
                    confidence = probabilities[1]
                    st.error(f"## **Prediction: {label} 🤖**")
                else:
                    label = "Human-Written"
                    confidence = probabilities[0]
                    st.success(f"## **Prediction: {label} 🧑‍💻**")
                
                st.metric(label="Confidence", value=f"{confidence:.2%}")
                st.progress(confidence)
                
                with st.expander("View Processed Text"):
                    st.code(processed_text, language=None)
        else:
            st.error("Model files are not loaded. Please check the logs.")
    else:
        st.info("Enter at least 250 characters and click 'Analyze Text' to see the results.")