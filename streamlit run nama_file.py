import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Title and Introduction
st.title('Weather Classification App')
st.write("This app predicts weather conditions based on the provided dataset and selected features.")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format):", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Basic Information
    st.write("### Dataset Information")
    st.write(data.describe())
    st.write("Data Shape:", data.shape)

    # Feature Selection
    st.write("### Feature Selection")
    features = st.multiselect("Select Features for Model Training:", options=data.columns, default=data.columns[:-1])
    target = st.selectbox("Select Target Column:", options=data.columns, index=len(data.columns)-1)

    if len(features) > 0 and target:
        X = data[features]
        y = data[target]

        # Split the dataset
        st.write("### Splitting Dataset")
        test_size = st.slider("Test Size (as a percentage):", 10, 50, 20, step=5) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        st.write("Training Set:", X_train.shape)
        st.write("Test Set:", X_test.shape)

        # Model Training
        st.write("### Model Training")
        n_estimators = st.slider("Number of Trees in Random Forest:", 10, 200, 100, step=10)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        st.write("Model trained successfully!")

        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"### Accuracy: {accuracy * 100:.2f}%")

        # Metrics
        st.write("### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)
