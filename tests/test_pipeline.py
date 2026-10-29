import unittest
import pandas as pd
import numpy as np
from src.processing.cleaning import load_and_clean_data
from src.processing.features import build_features

class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Create mini mock dataframes for testing
        self.mock_train_path = 'tests/mock_train.csv'
        self.mock_store_path = 'tests/mock_store.csv'
        
        train_data = {
            'Store': [1, 1, 1, 1, 1],
            'DayOfWeek': [5, 6, 1, 2, 3],
            'Date': ['2015-07-31', '2015-07-30', '2015-07-29', '2015-07-28', '2015-07-27'],
            'Sales': [5263, 6064, 8314, 0, 4822],
            'Customers': [555, 625, 821, 0, 559],
            'Open': [1, 1, 1, 0, 1],
            'Promo': [1, 1, 1, 0, 1],
            'StateHoliday': ['0', '0', '0', '0', '0'],
            'SchoolHoliday': [0, 0, 0, 0, 0]
        }
        pd.DataFrame(train_data).to_csv(self.mock_train_path, index=False)
        
        store_data = {
            'Store': [1],
            'StoreType': ['c'],
            'Assortment': ['a'],
            'CompetitionDistance': [1270.0],
            'CompetitionOpenSinceMonth': [9.0],
            'CompetitionOpenSinceYear': [2008.0],
            'Promo2': [0],
            'Promo2SinceWeek': [np.nan],
            'Promo2SinceYear': [np.nan],
            'PromoInterval': [np.nan]
        }
        pd.DataFrame(store_data).to_csv(self.mock_store_path, index=False)

    def tearDown(self):
        import os
        for f in [self.mock_train_path, self.mock_store_path]:
            if os.path.exists(f):
                os.remove(f)

    def test_cleaning_removes_closures_and_zero_sales(self):
        cleaned = load_and_clean_data(self.mock_train_path, self.mock_store_path)
        # Out of 5 rows, row 3 is Closed (Open=0, Sales=0) and should be removed
        self.assertEqual(len(cleaned), 4)
        self.assertTrue((cleaned['Sales'] > 0).all())
        self.assertTrue((cleaned['Open'] != 0).all())
        # Chronological sorting check
        self.assertTrue(cleaned['Date'].is_monotonic_increasing)
