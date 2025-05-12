import os
import pandas as pd




from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd

class DataframeAgent:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
        self.agent_executor = None
        self.responses = []

    def load_dataframe(self, df: pd.DataFrame):
        self.agent_executor = create_pandas_dataframe_agent(
            self.llm,
            df,
            agent_type="tool-calling",
            verbose=False,
            allow_dangerous_code=True
        )

    def chat(self, query: str) -> str:
        if not self.agent_executor:
            raise ValueError("Agent not initialized with a DataFrame.")
        response = self.agent_executor.invoke({"input": query})
        self.responses.append(response)
        return response["output"]

