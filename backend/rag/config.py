import os

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self,openai_api_key = None,domain_docs_dir=None,brand_docs_dir=None,chroma_persist_dir=None,domain_collection_name=None,brand_collection_name=None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.domain_docs_dir = domain_docs_dir or  "domain_docs/"
        self.brand_docs_dir = brand_docs_dir or "brand_docs/"
        self.data_csv_path = "data.csv"
        self.chroma_persist_dir = chroma_persist_dir or "chroma_db"  # Directory to persist ChromaDB
        self.domain_collection_name = domain_collection_name or "domain_knowledge"
        self.brand_collection_name = brand_collection_name or "brand_knowledge"
        
    def validate(self):
        """Validate configuration and required files"""
        if self.openai_api_key == "your-openai-api-key":
            raise ValueError("Please set your OpenAI API key in environment variable OPENAI_API_KEY")
        
        # Check if directories and files exist
        for path in [self.domain_docs_dir, self.brand_docs_dir, self.chroma_persist_dir]:
            if not os.path.exists(path):
                logger.warning(f"Directory {path} does not exist. Creating it...")
                os.makedirs(path, exist_ok=True)


if __name__=="__main__":
    Config()