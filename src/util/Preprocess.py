import numpy as np
import os
import time
import logging
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

np.random.seed(0)
# create logger
logger = logging.getLogger("main."+__name__)
dt_str_format = "%Y-%m-%d"
class Preprocess:
    def __init__(self):
        logging.info('Initializing Preprocessing')

    
    def process_high_cardinality_features(self,data,high_cardinality_features) :
        '''
        Preprocess and clean the process_high_cardinality_features

        Input : dataframe 
        Output : dataframe with lesser cardinality for HIGH CARDINALITY features
        
        '''
        logger.info('##### FUNCTION process_high_cardinality_features()')
        logger.info('Treat Education column to keep only the top 5 education ranks by counts')
        
        for i in high_cardinality_features : 
            logger.info('Treat {i} column to keep only the top 5 {i} by counts')
            top_5 = list(data.groupby(i)['user_id'].count().reset_index().sort_values(by='user_id',ascending=False).head(5).loc[:,i])

            data.loc[:,i] = data.loc[:,i].apply(lambda x : str(x) if x in top_5 else 'Other' )

    
        return data

    def preprocessing_pipeline(self,data,numeric_features,categorical_features):
        """
        Input : Raw Dataframe
        Output : Pre-processed dataframe
                    (Numerical and categorical features are manipulated appropriately)
        """
        logger.info('##### FUNCTION preprocessing_pipeline()')

        logger.info('##### Defining Numeric pipeline')
        numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
            ]
        )

        logger.info('##### Defining Categorical pipeline')
        categorical_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'unknown')),
                ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
            ]
            )

        logger.info('##### Defining ColumnTransformer')
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
                ]
                )

  
    
        clf = Pipeline(steps=[('preprocessor', preprocessor)
                            ])
        
        X =data.drop('great_customer_class', 1)
        y= data['great_customer_class']

        logger.info('##### Fitting the pre-processing pipeline')
        clf.fit(X,y)

       
        logger.info('##### Obtaining column names')
        col_names_cat=clf.named_steps['preprocessor'].transformers_[1][1].named_steps['onehotencoder'].get_feature_names(categorical_features)
        col_names = numeric_features + list(col_names_cat)
        logger.info(f'Column names are \n {col_names}')
        
        df_transformed =pd.DataFrame(clf.fit_transform(X),columns = col_names)
        logger.info(f'Shape of transformed dataframe is \n {df_transformed.shape}')

        return(df_transformed)
