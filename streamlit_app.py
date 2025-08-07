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

# --- Caching the Models ---
# This is a key Streamlit feature that prevents reloading the models on every interaction.
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
st.write("An advanced AI detector powered by a Random Forest model with **99.77% accuracy**.")

# --- Sidebar for Explanations ---
st.sidebar.header("About This App")
st.sidebar.write("""
This application uses a machine learning model to distinguish between human-written and AI-generated text.

**Workflow:**
1.  **Preprocessing:** The input text is cleaned using spaCy (lemmatization, stop word removal).
2.  **Vectorization:** The cleaned text is converted into numerical features using TF-IDF.
3.  **Prediction:** A pre-trained Random Forest model predicts the origin of the text.
""")
st.sidebar.info("The model was trained on a diverse dataset of over 300,000 text samples.")

# --- Input and Prediction ---
if model is None or vectorizer is None:
    st.error("Model files are not loaded. Please check the logs.")
else:
    input_text = st.text_area("Enter the text you want to analyze:", height=250, placeholder="Paste your text here...")

    if st.button("Analyze Text"):
        if input_text:
            with st.spinner("Analyzing..."):
                # Preprocess, vectorize, and predict
                processed_text = preprocess_text(input_text)
                vectorized_text = vectorizer.transform([processed_text])
                prediction = model.predict(vectorized_text)[0]
                probabilities = model.predict_proba(vectorized_text)[0]

                # Display results in columns for a cleaner look
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
                
                # Confidence meter
                st.progress(confidence)
        else:
            st.warning("Please enter some text to analyze.")