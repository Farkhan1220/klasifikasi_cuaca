import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Streamlit app title
st.title('Weather Classification App')

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)
    
    # Display dataset info
    st.subheader('Dataset Info')
    st.write(data.info())
    st.write(data.describe())
    
    # Display histogram of numerical features
    st.subheader('Numerical Feature Distributions')
    numerical_features = ['Suhu (\u00b0C)', 'Kelembapan (%)', 'Kecepatan Angin (km/jam)']
    for col in numerical_features:
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, bins=10, color='blue', ax=ax)
        ax.set_title(f'Distribution of {col}')
        st.pyplot(fig)
    
    # Display correlation matrix
    st.subheader('Correlation Matrix')
    fig, ax = plt.subplots()
    correlation_matrix = data[numerical_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)
    
    # Prepare data for modeling
    st.subheader('Model Training')
    X = data[numerical_features]
    y = data['Cuaca']
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train a RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Display classification report and confusion matrix
    st.subheader('Classification Report')
    st.text(classification_report(y_test, y_pred, target_names=le.classes_))
    
    st.subheader('Confusion Matrix')
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
