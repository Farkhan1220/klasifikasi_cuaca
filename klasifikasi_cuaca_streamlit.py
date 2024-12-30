import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Streamlit App Title
st.title("Weather Classification Analysis")

# Load the dataset
data = pd.read_csv("klasifikasi_cuaca.csv")

# Sidebar
st.sidebar.header("Settings")

def main():
    # Exploratory Data Analysis (EDA)
    st.subheader("Dataset Information")
    st.write(data.head())
    st.write("--- Dataset Info ---")
    st.text(data.info())
    st.write("--- Descriptive Statistics ---")
    st.write(data.describe())

    # Distribution Analysis
    st.subheader("Distribution Analysis")
    numerical_features = ['Suhu (°C)', 'Kelembapan (%)', 'Kecepatan Angin (km/jam)']
    for col in numerical_features:
        st.write(f"Distribution of {col}")
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, bins=10, ax=ax)
        st.pyplot(fig)

    # Encoding and Preprocessing
    st.subheader("Encoding and Preprocessing")
    label_encoder = LabelEncoder()
    data['Jenis Cuaca Encoded'] = label_encoder.fit_transform(data['Jenis Cuaca'])
    st.write("Encoded Weather Type:")
    st.write(data[['Jenis Cuaca', 'Jenis Cuaca Encoded']])

    # Feature Scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[numerical_features])
    st.write("Scaled Numerical Features:")
    st.write(pd.DataFrame(scaled_features, columns=numerical_features))

    # Splitting and Modeling
    st.subheader("Modeling and Evaluation")
    X = pd.DataFrame(scaled_features, columns=numerical_features)
    y = data['Jenis Cuaca Encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    st.write("--- Classification Report ---")
    report = classification_report(y_test, predictions, target_names=label_encoder.classes_, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    st.write("--- Confusion Matrix ---")
    conf_matrix = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
