pip install matplotlib seaborn pandas numpy scikit-learn streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Streamlit App Title
st.title("Klasifikasi Cuaca")

# File Upload
uploaded_file = st.file_uploader("Unggah file dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Dataset Overview
    st.subheader("Dataset Overview")
    st.write("### Informasi Dataset")
    st.write(data.info())
    st.write("### Statistik Deskriptif")
    st.write(data.describe())
    
    # Data Visualization
    st.subheader("Visualisasi Data")
    numerical_features = ['Suhu (°C)', 'Kelembapan (%)', 'Kecepatan Angin (km/jam)']
    
    if all(col in data.columns for col in numerical_features):
        # Histogram for numerical features
        for col in numerical_features:
            st.write(f"#### Distribusi {col}")
            fig, ax = plt.subplots()
            sns.histplot(data[col], kde=True, bins=10, color='blue', ax=ax)
            st.pyplot(fig)

        # Correlation Matrix
        st.write("#### Matriks Korelasi")
        fig, ax = plt.subplots(figsize=(8, 6))
        correlation_matrix = data[numerical_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Kolom yang dibutuhkan untuk visualisasi tidak ditemukan di dataset.")
    
    # Preprocessing
    st.subheader("Preprocessing")
    label_col = st.selectbox("Pilih kolom label", options=data.columns)
    if label_col:
        le = LabelEncoder()
        data[label_col] = le.fit_transform(data[label_col])
        st.write("### Label Encoding untuk kolom:", label_col)
        st.write(dict(zip(le.classes_, range(len(le.classes_)))))

        # Feature Selection
        features = [col for col in data.columns if col != label_col]
        X = data[features]
        y = data[label_col]

        # Splitting Dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model Training
        st.subheader("Model Training")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Model Evaluation
        st.subheader("Evaluasi Model")
        y_pred = model.predict(X_test)
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)
