---
title: Veritas AI - The Truth Detector
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.25.0
app_file: streamlit_app.py
pinned: false
---

# 🔎 Veritas AI: The Truth Detector

**Veritas AI** is an interactive application designed to determine the origin of a text, classifying it as either **Human-Written** or **AI-Generated**.

This app is powered by a **Random Forest** machine learning model that achieved **99.77% accuracy** on its test dataset. It was developed as a project for the NLP course by **Ahmed Aburakhia**.

---

## 🚀 How to Use the App

1.  **Enter Text:** Paste the text you want to analyze into the text box.
2.  **Check Length:** For the highest accuracy, please ensure your text is at least **250 characters** long. The character counter below the box will guide you.
3.  **Analyze:** Click the "Analyze Text" button.
4.  **View Results:** The app will display the prediction ("Human-Written" or "AI-Generated") along with the model's confidence score.

---

## 🛠️ About the Model

This application uses a classical machine learning pipeline to make its predictions:

-   **Preprocessing:** The input text is cleaned and processed using the **SpaCy** library.
-   **Feature Extraction:** The cleaned text is converted into numerical features using **TF-IDF**.
-   **Classification:** A **Random Forest** model, trained on a diverse dataset of over 300,000 text samples, makes the final prediction.

> **Note:** This model was trained on long-form essays. While highly accurate, its performance is best on texts that are paragraph-length or longer.

---

## 🔗 Project Links

-   **View the Code on GitHub:**