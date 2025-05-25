import logging
import chromadb
from langchain_openai import OpenAIEmbeddings, OpenAI
from typing import Optional
from langchain_chroma import Chroma
from dotenv import load_dotenv
from typing import List


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBManager:
    def __init__(self, persist_directory: str,api_key:str):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embeddings = OpenAIEmbeddings(model = "text-embedding-3-small",api_key=api_key)

    def add_documents_to_collection(self, documents: List, collection_name: str):
        """Add documents to ChromaDB collection or create it if not exists"""
        try:
            collection = None
            try:
                collection = self.client.get_collection(collection_name)
            except:
                collection = self.client.create_collection(collection_name)

            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                client=self.client,
                collection_name=collection_name
            )
            logger.info(f"Added {len(documents)} documents to collection '{collection_name}'")
            return vectorstore
        except Exception as e:
            logger.error(f"Error adding documents to collection '{collection_name}': {e}")
            return None

    def get_vectorstore(self, collection_name: str) -> Optional[Chroma]:
        """Load existing vectorstore"""
        try:
            return Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
        except Exception as e:
            logger.error(f"Error loading vectorstore for '{collection_name}': {e}")
            return None
