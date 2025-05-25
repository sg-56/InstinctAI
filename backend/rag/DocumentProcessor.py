from typing import List
from pathlib import Path
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import UploadFile
import tempfile
import shutil
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def ingest_file(self, file_path: str) -> List:
        """Load, split, and return documents from a file path"""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File does not exist: {file_path}")
            return []

        ext = path.suffix.lower()
        loader = None

        if ext == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        elif ext == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return []

        try:
            raw_docs = loader.load()
            return self.text_splitter.split_documents(raw_docs)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []

    async def ingest_upload_file(self, upload_file: UploadFile) -> List:
        """Load, split, and return documents from a FastAPI UploadFile"""
        suffix = Path(upload_file.filename).suffix.lower()

        if suffix not in {".txt", ".pdf"}:
            logger.warning(f"Unsupported file type: {suffix}")
            return []

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(upload_file.file, tmp)
                tmp_path = tmp.name

            return self.ingest_file(tmp_path)

        except Exception as e:
            logger.error(f"Error processing uploaded file {upload_file.filename}: {e}")
            return []
