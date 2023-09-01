import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from pymongo import MongoClient


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            def MongoDb_to_Dataframe():
    
                client = MongoClient("mongodb+srv://revankumar:revankumar@cluster0.pmcz5li.mongodb.net/")

                db = client['Adult_census']

                collection = db['collection']

                projection = {"_id": 0, "age": 1,"workclass": 1, "fnlwgt": 1,"education": 1, "education-num": 1, "marital-status": 1,
                "occupation": 1, "relationship": 1,"race": 1, "sex": 1,"capital-gain": 1, "capital-loss": 1,
                "hours-per-week": 1, "country": 1, "salary": 1}  # Replace with your field names

            

                # Retrieve data with specified projection
                data_from_db = collection.find({}, projection)

                Dataframe_from_db = pd.DataFrame(data_from_db)

                return Dataframe_from_db
                    
        
            
            def perform_random_oversampling():
    
        
                train_df = MongoDb_to_Dataframe()

                column_name_mapping = {'marital-status': 'marital_status', 'capital-gain': 'capital_gain', 'capital-loss': 'capital_loss',
                                        'hours-per-week':'hours_per_week','education-num':'education_num'}

                train_df = train_df.rename(columns=column_name_mapping)


                X = train_df.drop(columns=['salary'],axis=1)
                y = pd.DataFrame(train_df['salary'])
        
        
                ros = RandomOverSampler()
            
                X_resampled, y_resampled = ros.fit_resample(X, y)
        
                final_dataframe = pd.concat([X_resampled, y_resampled], axis=1)
        
                return final_dataframe
                
                
            
            df = perform_random_oversampling()
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df, test_size=0.3, random_state=30)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path)

        except Exception as e:
            
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    """modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_training(train_arr,test_arr))"""