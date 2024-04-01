import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import preprocessing_data, preprocessing_test_data, classifier_model, predict

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

    # Đọc dữ liệu huấn luyện từ file .csv
    dataset_file = st.file_uploader("Upload your dataset file", type=["csv"])
    if (dataset_file is None):
        st.stop()
    df = pd.read_csv(dataset_file)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Chọn các thuộc tính để huấn luyện
    features = st.multiselect(
        "Select features to train model",
        ['LOAIDICHVU', 'SODICHVU', 'SOLANBAOHONG', 'SOLANGOIKIEMBAOHONG', 'SOLANBAOHONGHAILONG', 'SOLANBAOHONGKHONGHAILONG', 'KHAOSATLAPMOI', 'KHAOSATLAPMOIHAILONG', 'KHAOSATLAPMOIKHONGHAILONG', 'SOLANCHAMSOC', 'SOLANTAMNGUNG', 'SOTHANGSUDUNGDICHVU', 'GIADICHVU', 'KHONGPHATSINHLUULUONG', 'DIEMTINNHIEM']
    )
            
    if (features):
        # Mã hóa tập huấn luyện
        st.subheader("Dataset after selected features")
        X_encoded = X.loc[:, features]
        X_encoded = preprocessing_data(X_encoded)
        with st.expander("Data preview"):
            st.dataframe(X_encoded)
                
        # Chọn giải thuật
        selected_ml_algorithms = st.multiselect(
            "Select ML algorithms",
            options=["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression"],
            default="K Nearest Neighbors"
        )
        
        # Huấn luyện các mô hình dựa trên các giải thuật đã chọn
        if st.button("Train", type="primary", use_container_width=True):
            cols = st.columns(len(selected_ml_algorithms))
            for col, ml_algorithm in zip(cols, selected_ml_algorithms):
                with col:
                    ml_model = classifier_model(X_encoded, y, ml_algorithm)
                    st.subheader(ml_algorithm)
                    st.write(f"Accuracy score: {ml_model[0]}")
                    st.write(f"F1 score: {ml_model[1]}")
                    
        st.header("Predict")

        # Đọc file dữ kiệu dự đoán
        input_file = st.file_uploader("Upload your input file", type=["csv"])
        if (input_file is None):
            st.stop()
        input_df = pd.read_csv(input_file)
        
        # Đưa ra dự đoán dựa trên các mô hình đã huấn luyện
        option = st.selectbox("Choose model was trained to predict", selected_ml_algorithms)
        if st.button("Predict", type="secondary"):
            temp = input_df.loc[:, features]
            temp = preprocessing_test_data(temp, X.loc[:, features])
            result = pd.DataFrame(predict(temp, option), columns=["THANHLY"])
            input_df = pd.concat([input_df.loc[:, features], result], axis=1)
            st.write(input_df)

if __name__ == "__main__":
    main()
