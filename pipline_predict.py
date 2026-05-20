import pandas as pd
import numpy as np
from gmpy2 import denom
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import datetime
import pickle
import joblib
import warnings


warnings.filterwarnings("ignore")


# input feature list
PREDICTION_INPUT_FEATURES = [
    'chain',
    'cereals',
    'milks',
    'population',
    'ms_by_city',
    'ms_by_chain_in_city',
    'aushan_stores_count',
    'detmir_stores_count',
    'lenta_stores_count',
]


# =========== custom transformers ===========

class InputFeaturesValidator(BaseEstimator, TransformerMixin):
    def __init__(self, required_features):
        self.required_features = required_features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        mising_features = set(self.required_features) - set(X.columns)
        if mising_features:
            raise ValueError(f'there are missing features: {", ".join(mising_features)}')
        return X


class FeatureCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        # create top_chains_total_count
        X_copy['top_chains_total_count'] = (X_copy['aushan_stores_count'] + 
                                            X_copy['lenta_stores_count'] + 
                                            X_copy['detmir_stores_count'])

        # create detmir_count_share_in_city
        X_copy['detmir_count_share_in_city'] = (
                X_copy['detmir_stores_count'] / 
                X_copy['top_chains_total_count'])

        # create lenta_count_share_in_city
        X_copy['lenta_count_share_in_city'] = (
                X_copy['lenta_stores_count'] / 
                X_copy['top_chains_total_count'])

        # create milks_cereals_ratio
        X_copy['milks_cereals_ratio'] = (
                (X_copy['milks'] + 1) / 
                (X_copy['cereals'] + 1))

        # create ms_by_chain_in_city
        chain_share = np.where(
            X_copy['chain'] == 'Ашан',
            X_copy['aushan_stores_count'] / X_copy['top_chains_total_count'].replace(0, 1),
            np.where(
                X_copy['chain'] == 'Лента',
                X_copy['lenta_stores_count'] / X_copy['top_chains_total_count'].replace(0, 1),
                np.where(
                    X_copy['chain'] == 'Детский мир',
                    X_copy['detmir_stores_count'] / X_copy['top_chains_total_count'].replace(0, 1),
                    0
                )
            )
        )
        X_copy['ms_by_chain_in_city'] = X_copy['ms_by_chain_in_city'] * chain_share
        #     'particular_store_share_in_city',
        #     'top_chains_stores_population_covered',
        #     'population_on_particular_store_share_in_city'
        
        return X_copy