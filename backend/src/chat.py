from dotenv import load_dotenv
import os
load_dotenv()
import pandas as pd

from src.s3 import S3Client


class ChatEngine:
    def __init__(self,API_KEY:str):
        import pandasai as pai
        import pandas as pd
        self.__api_key = API_KEY
        self.engine = pai
        self.engine.api_key.set(API_KEY)
        self.__data = None
        self.chat = None
        self.responses = []
        self.hasData = False

    def read_data(self,project_id:str):
        self.client = S3Client(bucket_name = os.getenv("BUCKET_NAME"),
        access_key  = os.getenv("ACCESS_KEY"),
        secret_key  = os.getenv("SECRET_KEY"),
        region_name = os.getenv("REGION"))
        data = self.client.getFile(project_id)
        data = pd.read_parquet(data)
        data.to_csv('data.csv',index=False)
        self.__data = data
        self.chat= self.engine.read_csv('data.csv')
        self.hasData = True
        os.remove('data.csv')
        
    def chat_with_data(self,query:str):
        response = self.chat.chat(query)
        self.responses.append(response.to_dict())
        return response.to_dict()

    def get_all_responses(self):
        return self.responses