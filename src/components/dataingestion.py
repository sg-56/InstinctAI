from dataclasses import dataclass
from pydantic import BaseModel
import pandas as pd
from collections import defaultdict


# @dataclass
class DataIngestion:
    data = None
    config = dict()
    constant_columns = None 
    def __init__(self):
        pass

    def ingest_from_file(self,file_path:str):
        if file_path is not None:
            self.config['source'] = file_path
            try:
                self.data = pd.read_csv(file_path)
                # return self.data
            except Exception as e:
                return f"Error reading file: {e}"
        else:
            return "No file path provided"      
        

    def ingest_from_db(self,db_connection_string:str,query:str):    
        if db_connection_string is not None:
            self.config.source = db_connection_string
            try:
                self.data = pd.read_sql_query(query,db_connection_string)
                return self.data
            except Exception as e:
                return f"Error reading db: {e}"
        else:
            return "No db connection string provided"

    def ingest_from_api(self,api_url:str):
        if api_url is not None:
            self.config.source = api_url
            try:
                self.data = pd.read_json(api_url)
                return self.data
            except Exception as e:
                return f"Error reading api: {e}"
        else:
            return "No api url provided"    
        

    def map_datatypes(self,schema:dict):
        if schema is not None:
            self.config.schema = schema
            try:
                self.data = self.data.astype(schema)
                return self.data
            except Exception as e:
                return f"Error mapping datatypes: {e}"
        else:
            return "No schema provided"
        
    def get_data(self):
        return self.data
    
    def get_column_dtypes(self):
        if self.data is not None:
            return self.data.dtypes.apply(lambda x: x.name).to_dict()
        else:
            raise ValueError("No data available.")
    

    def analyze(self):
        df = self.data

        # Identify missing values
        missing_values = {
            col: {
                "missing_count": count,
                **({"missing_indexes": df.index[df[col].isnull()].tolist()} if count <= 10 else {})
            }
            for col, count in df.isnull().sum().items() if count > 0
        }

        # Identify mixed data types in columns
        mixed_columns = {
            col: {"data_types": {str(t): round(pct, 2) for t, pct in self.dtype_ratio(df[col]).items()}}
            for col in df.columns if len(df[col].dropna().map(type).unique()) > 1
        }


        # Identify duplicate rows
        duplicate_rows = df[df.duplicated()]
        duplicate_row_count = duplicate_rows.shape[0]
        self.constant_columns = self.get_constant_columns()
        return {
            "columns_with_missing_values": missing_values,
            "mixed_data_types": mixed_columns,
            "constant_value_columns": self.constant_columns,
            "duplicate_rows": {
                "count": duplicate_row_count,
                "rows": duplicate_rows.to_dict(orient="records") if duplicate_row_count > 0 else []
            },
        }

    def get_constant_columns(self):
            """
            Returns a list of columns that have constant values.
            """
            if self.data is not None:
                constant_values = list({
                col: round((self.data[col].value_counts(dropna=False).iloc[0] / len(self.data[col])) * 100, 2)
                for col in self.data.columns if len(self.data[col].dropna()) > 0 and 
                (self.data[col].value_counts(dropna=False).iloc[0] / len(self.data[col])) * 100 >= 80
                }.keys())
                return constant_values
            else:
                return "Add data"
