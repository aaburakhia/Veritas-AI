# End-to-End AI-Generated Text Detector

This repository contains the source code and documentation for **Veritas AI**, an end-to-end text classification project to distinguish between human-written and AI-generated text. This project was completed for the **NLP Course**.

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/aaburakhia/Veritas-AI)

---

## Project Overview

The goal of this project was to build, implement, and evaluate a high-accuracy text classification system. The project involved two distinct workflows: a classical machine learning pipeline using Scikit-learn and a modern, integrated pipeline using SpaCy. The best-performing baseline model (Random Forest) was deployed as an interactive web application.

### Key Features:
-   **Data Pipeline:** Combines and preprocesses three large-scale datasets to create a balanced corpus of over 300,000 samples.
-   **Model Training:** Implements and evaluates two distinct models: a Random Forest Classifier and a SpaCy `TextCatBOW` model.
-   **High Accuracy:** The final models achieved an accuracy of **99.77% (Random Forest)** and **99.87% (SpaCy)** on a held-back test set.
-   **Live Demo:** The Random Forest model is deployed as an interactive Streamlit application on Hugging Face Spaces.

---

## Repository Structure
├── 1_Baseline_Models.ipynb # Jupyter Notebook for the Scikit-learn workflow 

├── 2_SpaCy_Model.ipynb # Jupyter Notebook for the SpaCy TextCatBOW workflow

└── huggingface_app/

├── streamlit_app.py # The Python code for the Streamlit application

├── requirements.txt # Python dependencies for the app

├── best_random_forest_model.joblib # The saved Random Forest model

└── tfidf_vectorizer.joblib # The saved TF-IDF vectorizer


---

## Methodology

### 1. Data Preprocessing
-   **Library:** SpaCy
-   **Tasks:** Tokenization, lemmatization, stop word removal, and punctuation removal.

### 2. Model Training & Evaluation
Two primary models were developed and compared:

| Model | Feature Extraction | Final Test Accuracy | Notebook |
| :--- | :--- | :--- | :--- |
| **Random Forest** | TF-IDF | **99.77%** | `1_Baseline_Models.ipynb` |
| **SpaCy `TextCatBOW`** | Integrated BoW | **99.87%** | `2_SpaCy_Model.ipynb` |

A rigorous train-validate-test split was used to ensure unbiased evaluation. Models were tuned using multiple hyperparameter configurations.

---

## Running the Application

The deployed application demonstrates the Random Forest model.

### Live Demo
The application is live on Hugging Face Spaces: **[Link to Your Veritas AI Hugg Face Space]**

### Local Setup
To run the Streamlit app locally:
1.  Clone this repository: `git clone [Your Repo URL]`
2.  Navigate to the `huggingface_app` directory: `cd huggingface_app`
3.  Install the required dependencies: `pip install -r requirements.txt`
4.  Run the Streamlit app: `streamlit run streamlit_app.py`

---

## Conclusion

This project successfully demonstrates the effectiveness of both classical and modern NLP techniques for AI text detection. The SpaCy-native pipeline achieved the highest performance, but the Random Forest model also proved to be exceptionally accurate and was successfully deployed as a practical, real-world application.```
