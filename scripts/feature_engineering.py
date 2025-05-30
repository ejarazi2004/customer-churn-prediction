import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineeringTransformer expects a DataFrame")
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineeringTransformer expects a DataFrame")        
        
        #Group based behavior features
        X['is_old_inactive'] = ((X['age'] > 50) & (X['active_member'] == 0)).astype(int)
        
        X['is_young_active'] = ((X['age'] < 30) & (X['active_member'] == 1)).astype(int)
        
        X['is_high_balance_short_tenure'] = ((X['balance'] > 100000) & (X['tenure'] < 3)).astype(int)
        
        X['is_low_product_inactive'] = ((X['products_number'] == 1) & (X['active_member'] == 0)).astype(int)
        
        X['is_german_customer'] = (X['country'] == 'Germany').astype(int)
        
        X['age_category'] = pd.cut(X['age'], bins=[0, 30, 50, 100], labels=['young', 'middle', 'old'])
        
        return X