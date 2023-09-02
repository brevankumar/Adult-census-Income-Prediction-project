import os
import sys
from dataclasses import dataclass

import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix


from imblearn.over_sampling import RandomOverSampler
from collections import Counter

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initate_model_training(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1])
            

            # Only the Random forest and Decision tree models given the higher accuracy as performed in the model training in jupyter notebook.
            # So, the Random forest and Decision tree models are considered.

            
            models = {
                "Random Forest": RandomForestClassifier(random_state=30),
                "Decision Tree": DecisionTreeClassifier(random_state=30),
                        }
            
            params={
                "Decision Tree": {
                    'criterion':["gini", "entropy"],
                    'max_depth': [2, 3, 5, 10, 20],
                    ##'min_samples_leaf': [5, 10, 20, 50, 100,150]
                },
                "Random Forest":{

                    'n_estimators': [int(x) for x in np.linspace(start=40, stop=150, num=15)],
                    'max_depth': [int(x) for x in np.linspace(40, 150, num=15)]
                    #'criterion':["gini", "entropy"],
                    #'min_samples_leaf': [5, 10, 20, 50, 100,150],
                    #'min_samples_split' : [2, 5, 10,14]
                }
                }
            
            
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                        models=models,param=params)
        
            print('\n==================================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy score : {best_model_score}')
            print('\n==================================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy score : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            with mlflow.start_run():

                predicted=best_model.predict(X_test)

                Accuracy_score = accuracy_score(y_test, predicted)
                F1_score       = f1_score (y_test, predicted)
                Roc_auc_score  = roc_auc_score(y_test, predicted)
        

                mlflow.log_metric("accuracy",Accuracy_score)
                mlflow.log_metric("roc_auc_score",Roc_auc_score)
                mlflow.log_metric("F1_score",F1_score)
                mlflow.log_param("n_estimators",150)
                mlflow.log_param("max_depth",10)
               # mlflow.log_param("criterion",'gini')
               # mlflow.log_param("min_samples_split", 5)
               # mlflow.log_param("min_samples_leaf",10)
                

                

                
            return Accuracy_score, X_train, y_train, X_test, y_test
            

            
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)
        

