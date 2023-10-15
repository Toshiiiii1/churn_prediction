import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_building import preprocessing_data, classifier_model, predict

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
        
    st.header("Train model")

    # Read data from csv file
    dataset_file = st.file_uploader("Upload your dataset file", type=["csv"])
    if (dataset_file is None):
        st.info("Upload a file through config", icon="ℹ️")
        st.stop()
    df = pd.read_csv(dataset_file)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Select features to train model from the dataset
    features = st.multiselect(
        "Select features to train model",
        ['LOAIDICHVU', 'SODICHVU', 'SOLANBAOHONG', 'SOLANGOIKIEMBAOHONG', 'SOLANBAOHONGHAILONG', 'SOLANBAOHONGKHONGHAILONG', 'KHAOSATLAPMOI', 'KHAOSATLAPMOIHAILONG', 'KHAOSATLAPMOIKHONGHAILONG', 'SOLANCHAMSOC', 'SOLANTAMNGUNG', 'SOTHANGSUDUNGDICHVU', 'GIADICHVU', 'KHONGPHATSINHLUULUONG', 'DIEMTINNHIEM']
    )
            
    if (features):
        # Encode data after selected features and display data
        st.subheader("Dataset after selected features")
        X_encoded = X.loc[:, features]
        X_encoded = preprocessing_data(X_encoded)
        with st.expander("Data preview"):
            st.dataframe(X_encoded)
                
        # Select ML algorithms
        selected_ml_algorithms = st.multiselect(
            "Select ML algorithms",
            options=["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression"],
            default="K Nearest Neighbors"
        )
        
        # Train data
        if st.button("Train", type="primary", use_container_width=True):
            cols = st.columns(len(selected_ml_algorithms))
            for col, ml_algorithm in zip(cols, selected_ml_algorithms):
                with col:
                    ml_model = classifier_model(X_encoded, y, ml_algorithm)
                    st.subheader(ml_algorithm)
                    st.write(f"Accuracy score: {ml_model[0]}")
                    st.write(f"F1 score: {ml_model[1]}")
                    
        st.header("Predict")
    
        input_file = st.file_uploader("Upload your input file", type=["csv"])
        if (input_file is None):
            st.stop()
        input_df = pd.read_csv(input_file)
        
        option = st.selectbox("Choose model was trained to predict", selected_ml_algorithms)
        
        if st.button("Predict", type="secondary"):
            result = predict(input_df, features, option)
            st.write(result)

if __name__ == "__main__":
    main()
