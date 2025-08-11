import streamlit as st
import joblib
import spacy
import pandas as pd
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Veritas AI | Advanced AI Detection",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Professional Styling ---
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    .stDeployButton {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom color scheme */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --error-color: #EF4444;
        --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main container styling */
    .main-container {
        background: white;
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* Header styling */
    .app-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2E86AB, #A23B72);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        color: #6B7280;
        font-weight: 400;
        margin-bottom: 1rem;
    }
    
    /* Input section styling */
    .input-section {
        margin: 2rem 0;
    }
    
    .char-counter {
        text-align: right;
        font-size: 0.875rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .char-counter.sufficient {
        color: var(--success-color);
    }
    
    .char-counter.insufficient {
        color: var(--warning-color);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2E86AB, #A23B72) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(46, 134, 171, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(46, 134, 171, 0.4) !important;
    }
    
    .stButton > button:disabled {
        background: #D1D5DB !important;
        color: #9CA3AF !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    /* Result cards */
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    .result-card.human {
        border-left-color: var(--success-color);
        background: linear-gradient(135deg, #ECFDF5, #F0FDF4);
    }
    
    .result-card.ai {
        border-left-color: var(--error-color);
        background: linear-gradient(135deg, #FEF2F2, #FEFEFE);
    }
    
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .confidence-display {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
    }
    
    .confidence-display.human {
        color: var(--success-color);
    }
    
    .confidence-display.ai {
        color: var(--error-color);
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background-color: rgba(0, 0, 0, 0.1);
    }
    
    /* Info section */
    .info-section {
        background: linear-gradient(135deg, #F8FAFC, #F1F5F9);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
        border: 1px solid #E2E8F0;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .feature-item {
        text-align: center;
        padding: 1rem;
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-title {
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.25rem;
    }
    
    .feature-desc {
        font-size: 0.875rem;
        color: #6B7280;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-container {
            padding: 1.5rem;
            margin: 1rem;
        }
        
        .app-title {
            font-size: 2rem;
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
MIN_CHARS = 250

# --- Caching the Models ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load('model/best_random_forest_model.joblib')
        vectorizer = joblib.load('model/tfidf_vectorizer.joblib')
        nlp = spacy.load("en_core_web_sm")
        return model, vectorizer, nlp
    except Exception as e:
        st.error(f"⚠️ Error loading model files: {e}")
        st.info("Please ensure the model files are properly uploaded to your Hugging Face Space.")
        return None, None, None

# --- Preprocessing Function ---
def preprocess_text(text):
    doc = nlp(text)
    processed_tokens = [
        token.lemma_.lower() 
        for token in doc 
        if not token.is_stop and not token.is_punct
    ]
    return " ".join(processed_tokens)

# Load models
model, vectorizer, nlp = load_models()

# --- Main Application ---
def main():
    # Header Section
    st.markdown("""
    <div class="main-container">
        <div class="app-header">
            <h1 class="app-title">🔍 Veritas AI</h1>
            <p class="app-subtitle">Advanced AI Detection Technology</p>
            <p style="color: #6B7280; font-size: 1rem;">Distinguish between human-written and AI-generated content with 99.77% accuracy</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown("""
        <div class="feature-grid">
            <div class="feature-item">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Lightning Fast</div>
                <div class="feature-desc">Get results in seconds</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">99.77% Accurate</div>
                <div class="feature-desc">Trained on diverse datasets</div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🔒</div>
                <div class="feature-title">Privacy First</div>
                <div class="feature-desc">No data stored or logged</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Input Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    # Text input with improved placeholder
    input_text = st.text_area(
        "📝 Enter your text for analysis",
        height=250,
        placeholder="Paste or type the text you want to analyze here. For best results, use at least 250 characters of content such as articles, essays, or detailed explanations...",
        key="user_input"
    )
    
    # Character count with better styling
    char_count = len(input_text.strip())
    
    if char_count < MIN_CHARS:
        st.markdown(f"""
        <div class="char-counter insufficient">
            📊 {char_count:,} / {MIN_CHARS:,} characters (minimum required)
        </div>
        """, unsafe_allow_html=True)
        st.warning(f"⚠️ Please provide at least {MIN_CHARS} characters for accurate analysis.")
    else:
        st.markdown(f"""
        <div class="char-counter sufficient">
            ✅ {char_count:,} / {MIN_CHARS:,} characters
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🔍 Analyze Text",
            use_container_width=True,
            disabled=(char_count < MIN_CHARS or not input_text.strip()),
            help="Click to analyze the text for AI detection" if char_count >= MIN_CHARS else f"Please provide at least {MIN_CHARS} characters"
        )
    
    # Results Section
    if analyze_button and model is not None and vectorizer is not None:
        with st.spinner("🔄 Analyzing your text... This may take a few seconds"):
            # Add a small delay for better UX
            time.sleep(1)
            
            try:
                # Process the text
                processed_text = preprocess_text(input_text)
                vectorized_text = vectorizer.transform([processed_text])
                prediction = model.predict(vectorized_text)[0]
                probabilities = model.predict_proba(vectorized_text)[0]
                
                # Determine result
                if prediction == 1:
                    label = "AI-Generated"
                    confidence = probabilities[1]
                    result_class = "ai"
                    emoji = "🤖"
                    color = "#EF4444"
                else:
                    label = "Human-Written"
                    confidence = probabilities[0]
                    result_class = "human"
                    emoji = "👤"
                    color = "#10B981"
                
                # Display results with enhanced styling
                st.markdown(f"""
                <div class="result-card {result_class}">
                    <div class="result-title">
                        <span>{emoji}</span>
                        <span>Detection Result: {label}</span>
                    </div>
                    <div class="confidence-display {result_class}">
                        {confidence:.1%} Confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence bar
                st.progress(confidence)
                
                # Additional insights
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Accuracy", "99.77%")
                with col2:
                    st.metric("📊 Characters", f"{char_count:,}")
                with col3:
                    st.metric("⚡ Processing", "< 2s")
                
                # Interpretation guide
                if confidence >= 0.9:
                    interpretation = "Very High - The model is highly confident in this prediction."
                    conf_color = "green"
                elif confidence >= 0.7:
                    interpretation = "High - The model is confident in this prediction."
                    conf_color = "blue"
                elif confidence >= 0.6:
                    interpretation = "Moderate - The prediction has reasonable confidence."
                    conf_color = "orange"
                else:
                    interpretation = "Low - The prediction should be interpreted with caution."
                    conf_color = "red"
                
                st.info(f"**Confidence Level:** {interpretation}")
                
                # Advanced details in expander
                with st.expander("🔬 Advanced Analysis Details"):
                    st.write("**Model Information:**")
                    st.write(f"- Model Type: Random Forest Classifier")
                    st.write(f"- Training Accuracy: 99.77%")
                    st.write(f"- Text Length: {char_count} characters")
                    st.write(f"- Processed Tokens: {len(processed_text.split())} words")
                    
                    st.write("**Probability Distribution:**")
                    prob_df = pd.DataFrame({
                        'Category': ['Human-Written', 'AI-Generated'],
                        'Probability': [probabilities[0], probabilities[1]]
                    })
                    st.bar_chart(prob_df.set_index('Category'))
                    
                    st.write("**Processed Text Sample:**")
                    st.code(processed_text[:200] + "..." if len(processed_text) > 200 else processed_text)
                
            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {str(e)}")
                st.info("Please try again with different text or contact support if the issue persists.")
    
    elif analyze_button and model is None:
        st.error("⚠️ Model files could not be loaded. Please check your deployment.")
    
    # Information Section
    if not analyze_button:
        st.markdown("""
        <div class="info-section">
            <h3 style="text-align: center; margin-bottom: 1rem;">How It Works</h3>
            <p style="text-align: center; color: #6B7280; margin-bottom: 1.5rem;">
                Our AI detection system uses advanced machine learning to analyze text patterns and linguistic features.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Use Streamlit columns instead of HTML grid for better compatibility
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📝</div>
                <strong>1. Text Analysis</strong><br>
                <small style="color: #6B7280;">We preprocess and analyze your text</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🤖</div>
                <strong>2. AI Processing</strong><br>
                <small style="color: #6B7280;">Our model evaluates linguistic patterns</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                <strong>3. Results</strong><br>
                <small style="color: #6B7280;">Get accurate predictions with confidence scores</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.875rem; margin-top: 2rem;">
        <p>🔍 Veritas AI - Advanced AI Detection Technology</p>
        <p>Built with Streamlit • Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar (Optional Information) ---
with st.sidebar:
    st.markdown("### 📊 About This Tool")
    st.info("""
    **Veritas AI** uses a Random Forest machine learning model trained on diverse text datasets to detect AI-generated content.
    
    **Key Features:**
    - 99.77% accuracy on test data
    - Real-time analysis
    - Privacy-focused (no data stored)
    - Works best with 250+ characters
    """)
    
    st.markdown("### 💡 Tips for Best Results")
    st.write("""
    - Use complete sentences or paragraphs
    - Minimum 250 characters recommended
    - Academic or professional writing works best
    - Avoid very short or fragmented text
    """)
    
    st.markdown("### 🚀 Model Performance")
    perf_data = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [99.77, 99.8, 99.7, 99.75]
    })
    st.dataframe(perf_data, hide_index=True)

if __name__ == "__main__":
    main()