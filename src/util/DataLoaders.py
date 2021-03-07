import logging
from abc import ABC, abstractmethod
import os.path
import errno
import os
import pandas as pd



class AbstractDataLoader(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_data(self, filename):
        logging.info('Checking file exists.')
        if not os.path.isfile(filename):
            logging.error('File does not exist')
            # TODO: raise exception

            print("file dosent exist")
        else:
            logging.info('Found file: ' + filename)

class FileDataLoader(AbstractDataLoader):

    # Initialization
    def __init__(self, filename: str):
       
        logging.info('Initializing Data Loading')
        self.filename = filename
        

    def load_data(self):
        """
        This method is used to load CSV data onto a dataframe and return the dataframe
        Input : Filename given while instantiating the class
        Output : Pandas DataFrame

        """
        ############ Check file exists #########
        logging.info('Checking file exists.')
        os.chdir('../data')
        if not os.path.isfile(self.filename):
            logging.error('File does not exist')
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.filename)
            # TODO: raise exception

            print("file dosent exist")
        else:
            logging.info('Found file: ' + self.filename)

        #############  Load data from file and return data ############ 
        logging.info('Loading data using pandas')
        df = pd.read_csv(self.filename)
        # TODO: Return your data object here
        return df

