import glob
import os
from rag.AgentManager import AgentManager
from rag.ResponseCoordinator import ResponseCoordinator
from typing import List,Dict
from rag.config import Config
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class ArchitectureRAGSystem:
    def __init__(self,config):
        self.config = config
        self.config.validate()
        self.agent_manager = AgentManager(self.config)
        self.coordinator = None

    def initialize(self,dataframe:pd.DataFrame):
        """Initialize agents with current Chroma collections and CSV"""
        domain_agent = self.agent_manager.create_retrieval_agent(self.config.domain_collection_name)
        brand_agent = self.agent_manager.create_retrieval_agent(self.config.brand_collection_name)
        table_agent = self.agent_manager.create_table_agent(self.config.dataframe)
        self.coordinator = ResponseCoordinator(domain_agent, brand_agent, table_agent)

    def ingest_file_to_domain(self, file_path: str):
        return self.agent_manager.ingest_and_index_file(file_path, self.config.domain_collection_name)

    def ingest_file_to_brand(self, file_path: str):
        return self.agent_manager.ingest_and_index_file(file_path, self.config.brand_collection_name)

    def query(self, question: str) -> str:
        if not self.coordinator:
            raise RuntimeError("System not initialized. Call initialize() first.")
        return self.coordinator.respond(question)

    def get_collection_info(self) -> Dict[str, int]:
        """Return document counts for collections"""
        info = {}
        try:
            for name in [self.config.domain_collection_name, self.config.brand_collection_name]:
                collection = self.agent_manager.chroma_manager.client.get_collection(name)
                info[name] = collection.count()
        except Exception as e:
            logger.error(f"Error fetching collection info: {e}")
        return info
    

if __name__ == "__main__":
    from dotenv import load_dotenv
    import pandas as pd
    df = pd.read_csv('../notebooks/data.csv')
    load_dotenv()
    config = Config(openai_api_key=os.getenv("OPENAI_API_KEY"),
                    domain_docs_dir="./notebooks/domain_docs",
                    brand_docs_dir="./notebooks/brand_docs",
                    chroma_persist_dir="./chroma_db"
                    )
    system = ArchitectureRAGSystem(config=config)
    query = input("Query : ")

    system.initialize(dataframe=df)
    response = system.query(query)
    print("*"*25)
    print(response)