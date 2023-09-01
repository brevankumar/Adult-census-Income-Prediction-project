import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
import numpy as np


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        


class CustomData:
    def __init__( self, age: int, workclass: str, education: str, education_num: int, marital_status: str, occupation: str,      
                 relationship: str, sex: str, capital_gain: int, capital_loss: int, hours_per_week: int,   
                  salary: str):
        self.age=age
        self.workclass= workclass
        self.education = education      
        self.education_num = education_num   
        self.marital_status = marital_status 
        self.occupation  = occupation    
        self.relationship  = relationship  
        self.sex =  sex         
        self.capital_gain = capital_gain   
        self.capital_loss = capital_loss   
        self.hours_per_week = hours_per_week  
        self.salary = salary
        
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age':[self.age],
                'workclass':[self.workclass],
                'education':[self.education],
                'education_num':[self.education_num],
                'marital_status':[self.marital_status],
                'occupation':[self.occupation],
                'relationship':[self.relationship],
                'sex':[self.sex],
                'capital_gain':[self.capital_gain],
                'capital_loss':[self.capital_loss],
                'hours_per_week':[self.hours_per_week],
                'salary':[self.salary]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        

        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)