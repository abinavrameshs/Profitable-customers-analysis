import unittest
import sys

# Below command is important to add the parent directory to system path
sys.path.append('../')
from  src.util.Preprocess import Preprocess
import os

import pandas as pd



class TestPreprocess(unittest.TestCase):

    def test_process_high_cardinality_features(self):
        Preprocess_obj = Preprocess()
        colors_list = ['red']*5 + ['blue']*3 + ['orange']*10 + ['green']*7 + ['pink']*1 + ['cyan']*4 + ['yellow']*2 + ['light blue']*6
        
        data = pd.DataFrame({'user_id': list(range(1,39)), 'colors':colors_list })
        
        
        data_return=Preprocess_obj.process_high_cardinality_features(data=data,high_cardinality_features=['colors'])


        # data_return should have just 6 categories

        self.assertEqual(data_return.groupby('colors')['user_id'].count().reset_index().shape[0],6)
        
       

if __name__ == '__main__':
    unittest.main()

