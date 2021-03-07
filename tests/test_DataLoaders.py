import unittest
import sys

# Below command is important to add the parent directory to system path
sys.path.append('../')
from src.util.DataLoaders import FileDataLoader
import os

import pandas as pd




class TestDataLoaders(unittest.TestCase):

    def test_load_data(self):
        data_loader = FileDataLoader(filename='final_dataset.csv') 
        data=data_loader.load_data()
        self.assertEqual(data_loader.load_data().shape,(13599,15))
        self.assertIsInstance(data, pd.DataFrame)
        
        with self.assertRaises(FileNotFoundError):
            data_loader1 = FileDataLoader(filename='garbage.csv') 
            data=data_loader1.load_data()

if __name__ == '__main__':
    unittest.main()

