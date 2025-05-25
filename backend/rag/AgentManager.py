from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chains import RetrievalQA
from rag.DocumentProcessor import DocumentProcessor
from rag.ChromaDBManager import ChromaDBManager
# import ChromaDBManager
import logging
from typing import Optional
import pandas as pd
# from . import Config

# config = Config()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentManager:
    def __init__(self, config):
        self.config = config
        self.llm = OpenAI(temperature=0)
        self.doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        self.chroma_manager = ChromaDBManager(config.chroma_persist_dir)

    def ingest_and_index_file(self, file_path: str, collection_name: str):
        """Process and ingest a single uploaded file"""
        docs = self.doc_processor.ingest_file(file_path)
        if not docs:
            logger.warning("No documents found to ingest.")
            return None
        return self.chroma_manager.add_documents_to_collection(docs, collection_name)

    def create_retrieval_agent(self, collection_name: str) -> Optional[RetrievalQA]:
        """Create a RetrievalQA chain for an existing ChromaDB collection"""
        vectorstore = self.chroma_manager.get_vectorstore(collection_name)
        if not vectorstore:
            logger.warning(f"Could not load vectorstore for '{collection_name}'")
            return None

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )

    def create_table_agent(self, dataframe: str | pd.DataFrame):
        try:
            if not isinstance(dataframe,pd.DataFrame):
                df = pd.read_csv(dataframe)
            
            return create_pandas_dataframe_agent(
                llm=self.llm,
                df=dataframe,
                verbose=True,
                allow_dangerous_code=True
            )
        except Exception as e:
            logger.error(f"Error loading CSV for table agent: {e}")
            return None

if __name__ == "__main__":
    print("Program called!")
