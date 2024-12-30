import pandas as pd
import numpy as np

# Streamlit App Title
st.title("Weather Classification Analysis")

# Sidebar for settings
st.sidebar.header("Settings")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload dataset (CSV format)", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("File successfully loaded!")

    # Dataset Information
    st.subheader("Dataset Information")
    st.write(data.head())
    st.text("Basic Info:")
    buffer = st.text_area("Info Output", data.info(buf=None), height=150)
    st.write("Descriptive Statistics:")
    st.write(data.describe())

    # Distribution Analysis
    st.subheader("Distribution Analysis")
    numerical_features = ['Suhu (°C)', 'Kelembapan (%)', 'Kecepatan Angin (km/jam)']
    for col in numerical_features:
        if col in data.columns:
            st.write(f"Distribution of {col}")
            fig, ax = plt.subplots()
            sns.histplot(data[col], kde=True, bins=10, ax=ax)
            st.pyplot(fig)

    # Encoding the Label
    st.subheader("Encoding the Label")
    label_col = st.sidebar.selectbox("Select Label Column", options=data.columns)
    if label_col:
        le = LabelEncoder()
        data[label_col] = le.fit_transform(data[label_col])
        st.write(f"Encoding applied to column: {label_col}")
        st.write(dict(zip(le.classes_, range(len(le.classes_)))))

        # Train-Test Split
        st.subheader("Train-Test Split")
        features = [col for col in data.columns if col != label_col]
        X = data[features]
        y = data[label_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write("Train and Test Split completed.")

        # Feature Scaling
        st.subheader("Feature Scaling")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        st.write("Feature scaling completed.")

        # Model Training
        st.subheader("Model Training")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)
        st.write("Model successfully trained.")

        # Model Evaluation
        st.subheader("Model Evaluation")
        y_pred = model.predict(X_test_scaled)
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

else:
    st.warning("Please upload a dataset to proceed.")
