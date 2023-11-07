import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, KFold
import pickle
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# tiền xử lý tập dữ liệu huấn luyện
def preprocessing_data(data):
    for feature in data.columns:
        # mã hóa các cột có dữ liệu rời rạc
        if (data[feature].dtype == 'object'):
            dummy = pd.get_dummies(data[feature], prefix=feature)
            data = pd.concat([data, dummy], axis=1)
            del data[feature]
        # chuẩn hóa các cột có dữ liệu liên tục
        elif (data[feature].dtype == 'int64'):
            scaler = MinMaxScaler()
            data[feature] = scaler.fit_transform(data[[feature]])
    return data

# tiền xử lý tập dữ liệu kiểm tra
def preprocessing_test_data(data, train_data):
    # ghép tập dữ liệu kiểm tra vào tập dữ liệu huấn luyện
    combine_data = pd.concat([train_data, data])
    test_len = len(data)
    for feature in combine_data.columns:
        # mã hóa các cột có dữ liệu rời rạc
        if (combine_data[feature].dtype == 'object'):
            dummy = pd.get_dummies(combine_data[feature], prefix=feature)
            combine_data = pd.concat([combine_data, dummy], axis=1)
            del combine_data[feature]
        # chuẩn hóa các cột có dữ liệu liên tục
        elif (combine_data[feature].dtype == 'int64'):
            scaler = MinMaxScaler()
            combine_data[feature] = scaler.fit_transform(combine_data[[feature]])
    # tách lấy tập dữ liệu kiểm tra
    return combine_data.tail(test_len)
    
# huấn luyện mô hình phân lớp
def classifier_model(X, y, ml_algorithm):
    classifier_algorithms = {
        "K Nearest Neighbors": KNeighborsClassifier(), 
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=1), 
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=1, max_depth=4), 
        "Logistic Regression": LogisticRegression(random_state=1),
    }
    
    # lấy mẫu SMOTE nhằm cân bằng tập dữ liệu
    smt = SMOTE()
    X_smote, y_smote = smt.fit_resample(X, y)
    
    # huấn luyện mô hình
    model = classifier_algorithms[ml_algorithm]
    model.fit(np.array(X_smote), y_smote)
    # lưu mô hình vào file .pkl
    pickle.dump(model, open(f"{ml_algorithm}.pkl", 'wb'))
    
    # đánh giá mô hình
    scoring = ("accuracy", "f1")
    k_fold = KFold(n_splits=10, random_state=0, shuffle=True)
    scores = cross_validate(model, np.array(X_smote), y_smote, scoring=scoring, cv=k_fold)
    accuracy = sum(scores['test_accuracy']) / len(scores['test_accuracy'])
    f1 = sum(scores['test_f1']) / len(scores['test_f1'])
    return (round(accuracy, 3), round(f1, 3))
    
# dự đoán
def predict(input_data, ml_algorithm):
    model = pickle.load(open(f"{ml_algorithm}.pkl", 'rb'))
    return model.predict(input_data)