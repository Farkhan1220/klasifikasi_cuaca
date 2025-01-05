import streamlit as st
import pandas as pd

# Define input fields based on the features used in your model
feature_1 = st.sidebar.number_input("Feature 1", min_value=0.0, max_value=100.0, value=50.0)
feature_2 = st.sidebar.number_input("Feature 2", min_value=0.0, max_value=100.0, value=50.0)
feature_3 = st.sidebar.number_input("Feature 3", min_value=0.0, max_value=100.0, value=50.0)

# Additional features can be added here

# Collect input features into a DataFrame
input_data = pd.DataFrame({
    'Feature 1': [feature_1],
    'Feature 2': [feature_2],
    'Feature 3': [feature_3]
})

# Display input data
st.write("### Input Data")
st.write(input_data)

# Preprocessing
scaler = StandardScaler()
scaled_input = scaler.fit_transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(scaled_input)
    st.write(f"### Predicted Weather Condition: {prediction[0]}")
