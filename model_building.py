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

class Model():
    
    # Encode data using One-hot method
    def encoding_data(data):
        for feature in data.columns:
            if (data[feature].dtype == 'object'):
                dummy = pd.get_dummies(data[feature], prefix=feature)
                data = pd.concat([data, dummy], axis=1)
                del data[feature]
        return data
    
    def classifier_model(X, y, ml_algorithm):
        classifier_algorithms = {
            "K Nearest Neighbors": KNeighborsClassifier(), 
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=0), 
            "Random Forest": RandomForestClassifier(n_estimators=10, random_state=0), 
            "Logistic Regression": LogisticRegression(random_state=0),
        }
        model = classifier_algorithms[ml_algorithm]
        scoring = ("accuracy", "f1_micro")
        k_fold = KFold(n_splits=10, random_state=0, shuffle=True)
        scores = cross_validate(model, np.array(X), y, scoring=scoring, cv=k_fold)
        accuracy = sum(scores['test_accuracy']) / len(scores['test_accuracy'])
        f1 = sum(scores['test_f1_micro']) / len(scores['test_f1_micro'])
        return (round(accuracy, 3), round(f1, 3))