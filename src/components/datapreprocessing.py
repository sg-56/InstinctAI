from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Tuple, Set, Dict
import warnings

class DataFramePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features=None, categorical_features=None, 
                 columns_to_drop=None,
                 impute_strategy_num='mean', impute_strategy_cat='most_frequent', scale_numerical=True):
        """
        Initialize the preprocessor
        
        Parameters:
        - numerical_features: list of numerical feature names
        - categorical_features: list of categorical feature names
        - columns_to_drop: list of columns to remove before processing
        - impute_strategy_num: strategy for numerical imputation ('mean', 'median', 'constant')
        - impute_strategy_cat: strategy for categorical imputation ('most_frequent', 'constant')
        - scale_numerical: whether to scale numerical features
        """
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.columns_to_drop = columns_to_drop or []
        self.impute_strategy_num = impute_strategy_num
        self.impute_strategy_cat = impute_strategy_cat
        self.scale_numerical = scale_numerical
        self.feature_names_ = None
        self.column_transformer_ = None
        self.feature_selector_ = None
        self.columns_removed_ = None
        self.important_features = None
        self.selected_features = None


    
    def setSelectedFeatures(self, selected_features: List[str]):
        """
        Set the selected features for the preprocessor.
        
        Parameters:
        - selected_features: list of feature names to select
        """
        self.selected_features = selected_features
    
    def fit(self, X, y=None):
        # Validate input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Remove specified columns

        self.columns_removed_ = [col for col in self.columns_to_drop if col in X.columns]
        X_processed = X.drop(columns=self.columns_removed_, errors='ignore')
            
        # If features not specified, use all remaining columns
        if self.selected_features is not None:
            columns = X_processed.columns.extend(x for x in self.user_features if x not in X_processed.columns)
            X_processed = X_processed[columns]
        
        if not self.numerical_features and not self.categorical_features:
            self.numerical_features = X_processed.select_dtypes(include=['number']).columns.tolist()
            self.categorical_features = X_processed.select_dtypes(exclude=['number']).columns.tolist()
            
        
        # Create transformers for numerical and categorical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.impute_strategy_num)),
            ('scaler', StandardScaler() if self.scale_numerical else 'passthrough')
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.impute_strategy_cat, fill_value='missing')),
            ('encoder', OrdinalEncoder())
        ])
        
        # Create column transformer
        self.column_transformer_ = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'
        )
        
        # Fit column transformer
        transformed_data = self.column_transformer_.fit_transform(X_processed)
        
        # Get feature names after transformation
        num_features = self.column_transformer_.named_transformers_['num'].get_feature_names_out(self.numerical_features)
        cat_features = self.column_transformer_.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(self.categorical_features)
        self.feature_names_ = np.concatenate([num_features, cat_features])
        self.is_fitted = True
        return self
        
    
    def transform(self, X):
        # Validate input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Remove specified columns
        X_processed = X.drop(columns=self.columns_removed_, errors='ignore')
        
        # Apply column transformations
        transformed_data = self.column_transformer_.transform(X_processed)
        
        # Apply feature selection if enabled
        if self.feature_selector_ is not None:
            transformed_data = self.feature_selector_.transform(transformed_data)
            
        # Return as DataFrame with feature names
        return pd.DataFrame(transformed_data, columns=self.feature_names_)
    
    def inverse_transform(self, X):
        """
        Inverse transform the processed data back to original space where possible.
        Note: Some transformations (like feature selection and column removal) are not perfectly invertible.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)
            
        # Handle feature selection inverse (approximate by padding with zeros)
        if self.feature_selector_ is not None:
            inverse_selected = np.zeros((X.shape[0], len(self.column_transformer_.get_feature_names_out())))
            selected_mask = self.feature_selector_.get_support()
            inverse_selected[:, selected_mask] = X.values
        else:
            inverse_selected = X.values
            
        # Inverse transform through column transformer
        result = self.column_transformer_.inverse_transform(inverse_selected)
        
        # Reconstruct DataFrame with original columns (including removed ones)
        original_columns = self.numerical_features + self.categorical_features
        reconstructed_df = pd.DataFrame(result, columns=original_columns)
        
        # Add back removed columns with NaN values
        for col in self.columns_removed_:
            reconstructed_df[col] = np.nan
            
        # Reorder columns to match original input
        all_columns = list(X.columns) + self.columns_removed_  # This might need adjustment
        return reconstructed_df.reindex(columns=all_columns, errors='ignore')

