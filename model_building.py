import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, KFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

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
    
def classifier_model(X, y, ml_algorithm):
    classifier_algorithms = {
        "K Nearest Neighbors": KNeighborsClassifier(), 
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=0), 
        "Random Forest": RandomForestClassifier(n_estimators=10, random_state=0), 
        "Logistic Regression": LogisticRegression(random_state=0),
    }
    model = classifier_algorithms[ml_algorithm]
    model.fit(np.array(X), y)
    pickle.dump(model, open(f"{ml_algorithm}.pkl", 'wb'))
    scoring = ("accuracy", "f1_micro")
    k_fold = KFold(n_splits=10, random_state=0, shuffle=True)
    scores = cross_validate(model, np.array(X), y, scoring=scoring, cv=k_fold)
    accuracy = sum(scores['test_accuracy']) / len(scores['test_accuracy'])
    f1 = sum(scores['test_f1_micro']) / len(scores['test_f1_micro'])
    return (round(accuracy, 3), round(f1, 3))
    
def predict(input_data, features, ml_algorithm):
    encoded_input_data = preprocessing_data(input_data.loc[:, features])
    model = pickle.load(open(f"{ml_algorithm}.pkl", 'rb'))
    return model.predict(encoded_input_data)