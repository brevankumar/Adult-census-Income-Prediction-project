import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer



#import category_encoders

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')



class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')


            # Categorigal Features

            workclass_categories = [' Never-worked', ' Without-pay',' Self-emp-inc',' Self-emp-not-inc',' Local-gov',' State-gov',' Federal-gov',' Private']
                        

            education_categories = [' Preschool',' 1st-4th',' 5th-6th', ' 7th-8th', ' 9th',' 10th',' 11th',' 12th',' Some-college',' HS-grad',
                                    ' Assoc-voc',' Assoc-acdm', ' Bachelors',' Prof-school',' Masters',' Doctorate']



            categorical_cols_for_OrdinalEncoding = ['workclass', 'education']


            categorical_cols_for_OnehotEncoding = ['marital_status', 'relationship', 'sex', 'occupation', 'race', 'country']


            # Numerical Features

            Numerical_cols  = ['age','education_num','capital_gain', 'capital_loss', 'hours_per_week']


            # Numerical Pipeline

            Numerical_pipeline_for_missingvalues = Pipeline(
                            steps=[
                            ('imputer',SimpleImputer(strategy='median')),
                            ('scaler',StandardScaler())
                            ])
            
            # Categorical pipeline
                        
            categorical_pipeline_for_OrdinalEncoding = Pipeline(
                            steps=[
                            ("imputer",SimpleImputer(strategy="most_frequent")),
                            ('OrdinalEncoder', OrdinalEncoder(categories=[workclass_categories,education_categories],dtype=int)),
                            ('scaler',StandardScaler())
                            ])

            categorical_pipeline_for_OnehotEncoding = Pipeline(
                            steps=[
                            ('imputer',SimpleImputer(strategy='most_frequent')),
                            ('one_hot_encoder', OneHotEncoder(sparse_output=False)),
                            ('scaler',StandardScaler())
                            ]
                            )


            preprocessor=ColumnTransformer(transformers= 
                    [('Numerical_pipeline_for_missingvalues', Numerical_pipeline_for_missingvalues,Numerical_cols),
                    ('cat_pipeline_for_OrdinalEncoding', categorical_pipeline_for_OrdinalEncoding,categorical_cols_for_OrdinalEncoding),
                    ('categorical_pipeline_for_OnehotEncoding', categorical_pipeline_for_OnehotEncoding,categorical_cols_for_OnehotEncoding),                      
                    ],
                                                                    remainder='passthrough',sparse_threshold=0)                 
                                                                    
                                                                    
                                                                    

            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
        

        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            salary_map = {' <=50K':0,' >50K':1}

            train_df['salary'] = train_df['salary'].map(salary_map)
            test_df['salary'] = test_df['salary'].map(salary_map)
            logging.info('transforming the salary column')

            train_df.replace(' ?',np.nan,inplace=True)
            test_df.replace(' ?',np.nan,inplace=True)
            logging.info('Replacing ? with NaN')

            logging.info(f'Train Dataframe head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe head: \n{test_df.head().to_string()}')

            target_column_name = 'salary'
            drop_columns = [target_column_name,'fnlwgt']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]

            ## Transformating using preprocessor obj

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
                










