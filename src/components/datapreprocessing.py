import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Optional
from sklearn.compose import ColumnTransformer
from shap import TreeExplainer
import shap
import numpy as np

class PreprocessingPipeline:
    def __init__(self, handle_missing: str = "mean", encode_categorical: bool = True, reverse_encoding: bool = False):
        self.handle_missing = handle_missing
        self.preprocessor = None
        self.encoders = {}
        self.best_features

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.handle_missing == "mean":
            return data.fillna(data.mean(numeric_only=True))
        elif self.handle_missing == "median":
            return data.fillna(data.median(numeric_only=True))
        elif self.handle_missing == "mode":
            return data.fillna(data.mode().iloc[0])
        else:
            raise ValueError("Unsupported missing value handling method")
    
    def encode_categorical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in data.select_dtypes(include=['object']).columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                data[col] = self.encoders[col].fit_transform(data[col])
            else:
                data[col] = self.encoders[col].transform(data[col])
        return data
    
    def reverse_encode(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in self.encoders:
            data[col] = self.encoders[col].inverse_transform(data[col])
        return data
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.handle_missing_values(data)
        if self.encode_categorical:
            data = self.encode_categorical_data(data)
        if self.reverse_encoding:
            data = self.reverse_encode(data)
        return data

    def select_best_features(self,data_frame:pd.DataFrame,target_column: str, k: int = 5):
        """
        Selects the best features
        """

        X = data_frame.drop(columns=[target_column])
        y = data_frame[target_column]
    
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        from sklearn.ensemble import RandomForestRegressor
        # from sklearn.feature_selection import SelectFromModel
        from sklearn.model_selection import train_test_split
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        # sfm = SelectFromModel(model, threshold=-np.inf, max_features=k)
        # sfm.fit(X_train, y_train)
        # selected_features = X_train.columns[sfm.get_support()]
        # return selected_features
        
    
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_valid)
    
    
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    
        feature_importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": mean_abs_shap
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
        return feature_importance_df.to_json()
        

    def describe_columns(self) -> pd.DataFrame:
        """
        Returns a description of each column including data types and unique counts.
        """
        description = pd.DataFrame({
            'Column': self.data.columns,
            'Data Type': self.data.dtypes.values,
            'Unique Values': [self.data[col].nunique() for col in self.data.columns],
            'Missing Values': self.data.isnull().sum().values
        })
        return description

    def prepare_for_clustering(self) -> pd.DataFrame:
        """
        Prepares data for clustering by normalizing numeric columns.
        """
        numeric_columns = self.data.select_dtypes(include=['number']).columns
        self.data[numeric_columns] = (self.data[numeric_columns] - self.data[numeric_columns].mean()) / self.data[numeric_columns].std()
        return self.data
        
    def prepare_for_time_series(self, time_column: str) -> pd.DataFrame:
        """
        Prepares data for time series analysis by setting the index to the time column.
        """
        if time_column not in self.data.columns:
            raise ValueError("Time column not found in data.")
        self.data[time_column] = pd.to_datetime(self.data[time_column])
        self.data.set_index(time_column, inplace=True)
        return self.data
        