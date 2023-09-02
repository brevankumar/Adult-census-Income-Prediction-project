import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
import mlflow
import mlflow.sklearn

from src.logger import logging


from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            # Random Search Cross Validation

            skfold=StratifiedKFold(n_splits=5)
            

            Randomized_Search = RandomizedSearchCV(estimator= model, param_distributions=para, cv=skfold,random_state=30)
            Randomized_Search.fit(X_train, y_train)

            logging.info(f'Best parameters are: {Randomized_Search.best_params_}')


            model.set_params(**Randomized_Search.best_params_,random_state=30)
            model.fit(X_train,y_train)


            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)