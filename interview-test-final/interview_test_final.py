import logging
import json
from collections import namedtuple
from util import DataLoaders,Predictors,Preprocess
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.svm import SVC
import time

def sort_file_paths(project_name: str):
    # figure out the path of the file we're runnning
    """
    #runpath = os.path.realpath(__file__)
    runpath =os.path.realpath(os.getcwd())
    # trim off the bits we know about (i.e. from the root dir of this project)
    rundir = runpath[:runpath.find(project_name) + len(project_name) + 1]
    # change directory so we can use relative filepaths
    os.chdir(rundir + project_name)
    """
    os.chdir(os.getcwd())

def load_config():
    run_configuration_file = '../resources/interview-test-final.json'
    with open(run_configuration_file) as json_file:
        json_string = json_file.read()
        run_configuration = json.loads(json_string,
                                       object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    return run_configuration

if __name__ == '__main__':
    start_time = time.time()
    # Initialize logging
    logging.basicConfig(format="%(asctime)s;%(levelname)s;%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logging.info('Starting classification program')

    # Actions: get into working directory, load project config, create dated directories
    sort_file_paths(project_name='interview-test-final')
    run_configuration = load_config()


    # TODO: Load the data by instantiating the FileDataLoader, handle file doesn't exist.
    # Candidate , instantiate your class here
    
    data_loader = DataLoaders.FileDataLoader(filename=run_configuration.file_name) 
    data=data_loader.load_data()

    logging.info(f'Printing some data,\n {data.head()}')

    # TODO: Do the rest of your work here, or in other classes that are called here.
    logging.info(f' #####################################')
    logging.info(f' ##### Preprocessing of data starts #####')
    logging.info(f' #####################################')

    logging.info(f' ##### Remove duplicates #####')
    data.drop_duplicates(inplace=True)
    logging.info(f'Shape after removing duplicates =\n {data.shape}')
    logging.info(f' ##### Process Variables appropriately\n Defining different kinds of variables #####')

    remove_features = ['user_id']
    numeric_features = ['age','salary','mins_beerdrinking_year','mins_exercising_year','works_hours','tea_per_year','coffee_per_year']
    high_cardinality_features = ['education_rank','occupation']
    categorical_features = ['workclass','marital-status','race', 'sex','education_rank','occupation']

    logging.info(f' ##### Defining Pre-processing object #####')
    Preprocess_obj = Preprocess.Preprocess()
    
    data = Preprocess_obj.process_high_cardinality_features(data=data,high_cardinality_features = high_cardinality_features)

    logging.info(f'Drop user ID column')
    data = data.drop('user_id',axis=1)
    logging.info(f'Shape after dropping USER_ID =\n {data.shape}')
    
    logging.info(f'Split into X and y variables')
    X = data.loc[:,~data.columns.isin(['great_customer_class'])]
    y= data['great_customer_class']

    print(f"Shape of X is  {X.shape} and the length of Y is {len(list(y))} ")
    print(f"####### Calling Preprocessing pipeline ###### ")
    X = Preprocess_obj.preprocessing_pipeline(data=data,numeric_features=numeric_features,categorical_features=categorical_features)
    
    logging.info(f' #####################################')
    logging.info(f' ##### Preprocessing Ends #####')
    logging.info(f' #####################################')

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)
    logging.info(f'Split into Test and train with test set = 20% and training set = 80%')
   
    logging.info(f' #####################################')
    logging.info(f' ##### Apply Synthetic Minority oversampling only to Training data #####')
    logging.info(f' #####################################')
    
    smote = SMOTE(random_state=0)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    logging.info(f'Shape of X_train after SMOTE  =\n {X_train.shape}\n Shape of y_train after SMOTE \n {len(list(y_train))}')

    logging.info(f' #####################################')
    logging.info(f' ##### Start Modelling #####')
    logging.info(f' #####################################')
    # Candidate, instantiate your class here
    logging.info(f' #####################################')
    logging.info(f' ##### Running Random Forest Model #####')
    logging.info(f' #####################################')
    
    rf_model=Predictors.Run_Model(kind='rf')
    rf_model.train(X_train=X_train,y_train=y_train)
    y_pred=rf_model.predict(X_test=X_test)
   
    logging.info(f"length of y_actual is {len(list(y_test))} ")
    #print("y_actual is ,", list(y_test))
    performance_metrics_rf = rf_model.get_metrics(y_pred=y_pred,y_actual=y_test)
    logging.info(f"performance_metrics_rf is \n {performance_metrics_rf} ")

    logging.info(f' #####################################')
    logging.info(f' ##### Obtain Dominant features #####')
    logging.info(f' #####################################')
    
    df_features=rf_model.important_features(X_train=X_train)
    
    logging.info(f' Dominant features are \n {df_features} #####')
    
    logging.info(f' #####################################')
    logging.info(f' ##### Running an SVM model #####')
    logging.info(f' #####################################')
    

    svc_model = Predictors.Run_Model(kind='SVC')
    svc_model.train(X_train=X_train,y_train=y_train)
    y_pred_svc=svc_model.predict(X_test=X_test)
    performance_metrics_svc = svc_model.get_metrics(y_pred=y_pred_svc,y_actual=y_test)
    logging.info(f"performance_metrics_svc is \n {performance_metrics_svc} ")

    T = time.time() - start_time
    logging.info(f'TOTAL RUNTIME of the Program = {T} secs')
    logging.info('Completed program')
