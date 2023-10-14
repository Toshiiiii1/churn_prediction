import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_building import Model

st.set_page_config(
    page_title="Churn prediction",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    st.write("""
        # Customer Churn Prediction App

        This application predicts **customer churn** from telecommunications services
    """)
    
    
    st.sidebar.header('User Input Features')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        
    st.header("Train model and predict")

    # Read data from csv file
    df = pd.read_csv("D:\Code\Python Code\data\customer-v2.csv")
    df.loc[df.PHUONGXA == "Thị Trấn Phong Điền", "PHUONGXA"] = "TT Phong Điền"

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Select features to train model from the dataset
    features = st.multiselect(
        "Select features to train model",
        X.columns
    )
        
    if (features):
        # Encode data after selected features and display data
        st.subheader("Dataset after selected features")
        X_encoded = X.loc[:, features]
        X_encoded = Model.encoding_data(X_encoded)
        st.dataframe(X_encoded)
            
        # Select ML algorithms
        selected_ml_algorithms = st.multiselect(
            "Select ML algorithms",
            options=["K Nearest Neighbors", "Naive Bayes", "Decision Tree", "Random Forest", "Logistic Regression"],
            default="K Nearest Neighbors"
        )
            
        # Train data
        if st.button("Train", type="primary", use_container_width=True):
            cols = st.columns(len(selected_ml_algorithms))
            for col, ml_algorithm in zip(cols, selected_ml_algorithms):
                with col:
                    temp = Model.classifier_model(X_encoded, y, ml_algorithm)
                    st.subheader(ml_algorithm)
                    st.write(f"Accuracy score: {temp[0]}")
                    st.write(f"F1 score: {temp[1]}")
                    st.button("Predict", type="secondary", key=ml_algorithm)

if __name__ == "__main__":
    main()
