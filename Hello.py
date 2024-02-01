import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

# Create a file upload button
uploaded_file = st.file_uploader("Choose a file to upload:", type="pkl")

# If a file is uploaded, load the ML model
if uploaded_file is not None:
    try:
        # Load the ML model directly from the BytesIO object using joblib
        model = joblib.load(uploaded_file)

        # Check if the model is an LGBMClassifier model
        if not isinstance(model, LGBMClassifier):
            raise TypeError("The uploaded model is not an LGBMClassifier model.")

        # Get the names of the variables in the dataset
        var_names = model.feature_name_

        # Create a box to show model information
        st.header("Model Information")
        st.write(f"Model Parameters: {model.get_params()}")
        st.write(f"Features: {var_names}")

        # Create sliders for each selected variable
        new_data = {}
        for var_name in var_names:
            point_val = st.slider(f'Select a point for {var_name} within the range:', 0, 10, 10)
            new_data[var_name] = point_val

        # Create a button to predict the target
        if st.button('Predict'):
            # Create the new dataset with the selected variables and their values
            new_df = pd.DataFrame([new_data], index=[0])  # Specify an index

            # Predict the target
            prediction = model.predict(new_df)

            # Display the prediction
            st.write(f"Prediction: {prediction[0]}")

    except Exception as e:
        st.write(f"Error loading the model: {e}")
else:
    st.write("Please upload an ML model file (.pkl)")
