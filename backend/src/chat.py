import pandas as pd
import os
class ChatEngine:
    def __init__(self,API_KEY:str):
        import pandasai as pai
        from pandasai_openai import OpenAI
        import pandas as pd
        self.engine = pai
        llm = OpenAI(api_token=API_KEY)
        self.engine.config.set({"llm": llm})
        self.__data = None
        self.chat = None
        self.responses = []
        self.hasData = False





    def read_data(self,s3,project_id:str):
        data = s3.getFile(project_id)
        data = pd.read_parquet(data)
        data.to_csv('data.csv',index=False)
        self.__data = data
        self.chatengine = self.engine.read_csv('data.csv')
        self.hasData = True
        os.remove('data.csv')




    def chat_with_data(self,query:str):
        response = self.chatengine.chat(query)
        self.responses.append(response)
        if response.type == 'dataframe':
            self.responses.append(response.to_dict())
            return response.to_dict()
        if response.type == 'number' or response.type == 'string':
            return response.value
        if response.type == 'chart':
            return response.get_base64_image()


    def get_all_responses(self):
        return self.responses

