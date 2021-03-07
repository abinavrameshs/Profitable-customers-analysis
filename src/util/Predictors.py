import logging
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import pandas as pd
from sklearn.svm import SVC
import logging

logging.basicConfig(format="%(asctime)s;%(levelname)s;%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logging.info('Starting classification program')
logger = logging.getLogger("main."+__name__)

class Model(ABC):

    def __init__(self):
        super().__init__()
        logger.info('Initializing model')

    # TODO: Feel free to add or change these methods.
    @abstractmethod
    def train(self):
        logger.info('Training model')

    @abstractmethod
    def predict(self):
        logger.info('Doing predictions')


class Run_Model(Model):
    
    def __init__(self,kind):
        super().__init__()
        #self.kind = kind
        logger.info('Initializing model')
        self.model=None
        if(kind=='rf'):
            self.model = RandomForestClassifier(random_state=0)
            logger.info('Initialized a basic Random Forest model')
        if(kind=='SVC'):
            self.model = SVC(gamma='auto',random_state=0)
            logger.info('Initialized a basic SVC model')
            

    def train(self,X_train,y_train):
        """
        This method is used to run the fit method on any model
        Input : X_train, y_Train
        Output : No output, but the model with be fitted with X_train and y_Train
        """
        logger.info('FUNCTION train()')
        logger.info('Fitting/Training model')
        self.model.fit(X_train,y_train)
       

    def predict(self,X_test):
        """
        This method is used to run the predict method on any model
        Input :X_tets
        Output :Predictions on the test set of any model
        """
        logger.info('FUNCTION predict()')
        logger.info('Performing and returning predictions')
       
        logger.info(f'Length of predictions is {len(list(self.model.predict(X_test)))}')
        return self.model.predict(X_test)

    def get_metrics(self,y_pred,y_actual):
        """
        This method provides the performance metris given the predictions and the actual outcomes
        Input : predictions, actuals
        Output : Performance metrics (Accuracy, F1, AUC, precision, recall)
        """
        logger.info('FUNCTION get_metrics()')
        accuracy = accuracy_score(y_actual, y_pred)
        prec = precision_score(y_actual, y_pred)
        recall = recall_score(y_actual, y_pred)
        f1 = f1_score(y_actual, y_pred)
        auc = roc_auc_score(y_actual, y_pred)
        
        data = {
            'accuracy' : [accuracy],
            'prec' : [prec],
            'recall': [recall],
            'f1' : [f1],
            'auc' : [auc]
        }

        return pd.DataFrame.from_dict(data)

    def important_features(self,X_train):
        """
        This method provides is exclusively used for any Ensemble-tree based models that are capable of providing feature importances

        Input : X_train
        Output : Feature importance dataframe
        """
        logger.info('FUNCTION important_features()')
        feature_importances = pd.DataFrame(self.model.feature_importances_,
                                                    index = X_train.columns,columns=['importance']).sort_values('importance',ascending=False)
        df_feature=feature_importances.head(15).reset_index().rename(columns={'index': 'feature'})
        return(df_feature)

    
    


