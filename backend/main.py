import json
import smtplib
import random
import uuid
from uuid import uuid4
import bcrypt
from fastapi import FastAPI, HTTPException, Query, Path, Body, UploadFile, File,Response
from fastapi.responses import JSONResponse,HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import redis
from typing import List, Optional, Union
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
import os
import string
# import io
# from io import BytesIO
import pandas as pd
from src.s3 import S3Client
from src.components.dataingestion import DataIngestion
from src.components.datapreprocessing import DataFramePreprocessor
from src.components.clustering import ClusteringEngine
from rag.config import Config
from rag.AgentManager import AgentManager
from rag.ChromaDBManager import ChromaDBManager
from rag.DocumentProcessor import DocumentProcessor
from rag import ArchitectureRAGSystem
# from src.utils import reduce_memory_usage
from datetime import datetime
import io
import pyarrow.parquet as pq
import requests 
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()
app = FastAPI(title="Architecture RAG System" , swagger_ui_parameters={"syntaxHighlight": False})


# ✅ CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

s3 = S3Client(
    bucket_name = os.getenv("BUCKET_NAME"),
    access_key  = os.getenv("ACCESS_KEY"),
    secret_key  = os.getenv("SECRET_KEY"),
    region_name = os.getenv("REGION")
)

SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = os.getenv("SMTP_PORT")
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")


ingestor  = DataIngestion()
engine    = ClusteringEngine()

rag_config = Config(
    openai_api_key=os.getenv("OPEN_AI_KEY"),
    brand_docs_dir="./brand_docs",
    domain_docs_dir="./domain_docs",
    chroma_persist_dir="./chroma_db",
    domain_collection_name="domain_knowledge",
    brand_collection_name="brand_knowledge"
)

rag_config.validate()

docs_processor = DocumentProcessor()
chroma_db = ChromaDBManager(persist_directory=rag_config.chroma_persist_dir,api_key=rag_config.openai_api_key)
rag_system = ArchitectureRAGSystem(config=rag_config)



# 📦 Redis Setup
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

# 🔐 Session Expiry Time (in seconds)
SESSION_EXPIRE_TIME = 86400  # 24 hours


# 📦 MongoDB Atlas Connection
mongo_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongo_uri)
db = client["my-database2"]
users_col = db["users"]
projects_col = db["projects"]



# ✅ Schemas
class OTPRequest(BaseModel):
    email: EmailStr

class OTPVerify(BaseModel):
    email: EmailStr
    otp: str

class SetPasswordModel(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class LogoutRequest(BaseModel):
    session: str

class ColumnList(BaseModel):
    columns: List[str]
    class Config:
        frozen = True

class ProjectModel(BaseModel):
    com_id: str
    name:str
    description: str
    columns: List[str]
    importantColumnNames: List[str]
    kpiList: List[str]
    droppedColumns: List[str]
    uploadedFileData: str
    selectedKpi: Optional[str]


class NewProject(BaseModel):
    name:str
    description:str

class ClusterStep(BaseModel):
    clusterIndex: int
    feature: str
    value: Union[str, int]

class ClusterJourneyUpdate(BaseModel):
    project_id: str
    cluster_journey: List[ClusterStep]
    cluster_selection_index: int


# ✅ Email Function
def send_otp(email):
    otp = str(random.randint(100000, 999999))
    message = f"Subject: Your OTP Code\n\nYour OTP is: {otp}"

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.sendmail(SMTP_EMAIL, email, message)
        server.quit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")

    return otp

def generate_com_id(length=8):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))


# ✅ OTP Request Endpoint
@app.post("/signup/request-otp")
def signup_request_otp(data: OTPRequest):
    user = users_col.find_one({"email": data.email})
    if user and user.get("verified") == 1:
        raise HTTPException(status_code=400, detail="User already verified")

    if not user:
        users_col.insert_one({"email": data.email, "verified": 0})

    otp = send_otp(data.email)
    redis_client.setex(f"otp:{data.email}", 300, otp)
    return {"message": "OTP sent successfully", "otp": otp}

# ✅ OTP Verification
@app.post("/signup/verify-otp")
def verify_otp(data: OTPVerify):
    stored_otp = redis_client.get(f"otp:{data.email}")
    if stored_otp != data.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    user = users_col.find_one({"email": data.email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Generate com_id if not already present
    if "com_id" not in user:
        com_id = generate_com_id()
        # Ensure uniqueness
        while users_col.find_one({"com_id": com_id}):
            com_id = generate_com_id()
    else:
        com_id = user["com_id"]

    users_col.update_one(
        {"email": data.email},
        {"$set": {"verified": 1, "com_id": com_id}}
    )
    redis_client.delete(f"otp:{data.email}")
    return {"message": "OTP verified. Now you can set your password."}


# ✅ Set Password
@app.post("/signup/set-password")
def set_password(data: SetPasswordModel):
    user = users_col.find_one({"email": data.email})
    if not user or user.get("verified") != 1:
        raise HTTPException(status_code=400, detail="Email not verified yet")

    hashed_password = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()
    users_col.update_one({"email": data.email}, {"$set": {"password": hashed_password}})

    updated_user = users_col.find_one({"email": data.email})
    com_id = str(updated_user["_id"])
    return {"message": "Password set successfully", "com_id": com_id}

# ✅ Login Route
@app.post("/login")
def login(user: UserLogin):
    db_user = users_col.find_one({"email": user.email})
    if not db_user or not bcrypt.checkpw(user.password.encode(), db_user.get("password", "").encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if db_user.get("verified") != 1:
        raise HTTPException(status_code=403, detail="Email not verified")

    com_id = str(db_user["com_id"])
    session_token = str(uuid.uuid4())
    redis_client.setex(f"session:{session_token}", SESSION_EXPIRE_TIME, com_id)

    has_projects = projects_col.count_documents({"com_id": com_id}) > 0
    return {
        "message": "Login successful",
        "com_id": com_id,
        "session": session_token,
        "has_projects": has_projects
    }

# ✅ Logout
@app.post("/logout")
def logout(data: LogoutRequest):
    session_key = f"session:{data.session}"
    if not redis_client.exists(session_key):
        raise HTTPException(status_code=401, detail="Invalid session or already logged out")

    redis_client.delete(session_key)
    return {"message": "Logged out successfully"}

# ✅ Get Projects
@app.get("/{com_id}/projects/")
def get_projects(com_id: str = Path(...)):
    try:
        rows = projects_col.find(
            {"com_id": com_id},
            {
                "name": 1,
                "description": 1,
                "project_id": 1,
                "created_at": 1,  # ✅ Include this
                "_id": 0
            }
        )
        projects = [{"name": row.get("name", ""), 
                     "description": row.get("description", ""),
                     "project_id": row.get("project_id"),
                     "created_at": row.get("created_at", "")
                     } for row in rows]

        return {"projects": projects}
    except Exception as e:
        raise HTTPException(500, detail=f"Error fetching projects: {e}")

# ✅ Get Projects
# Added project_id to the response by shankersingh01
@app.get("/{com_id}/projects/")
def get_all_projects(com_id: str = Path(...)):
    try:
        rows = projects_col.find(
            {"com_id": com_id},
            {"project_id": 1, "name": 1, "description": 1, "_id": 0}
        )

        projects = [
            {
                "project_id": row.get("project_id", ""),
                "name": row.get("name", ""),
                "description": row.get("description", "")
            }
            for row in rows
        ]

        return {"projects": projects}
    except Exception as e:
        raise HTTPException(500, detail=f"Error fetching projects: {e}")
    

# @app.get("/{com_id}/projects/{project_id}")
# def get_project_details(com_id:str = Path(...),
#                         project_id:str = Path(...)):
#     try : 
#         result = projects_col.find(
#                 {"com_id": com_id,
#                 "project_id": project_id,
#                 },
#                 {'_id': 0}
#         )
#         keys = list([ row  for row in result])
#         return keys[0]
    
#     except Exception as e :
#         raise HTTPException(500,detail=f"Error Fetching projects: {e}")

@app.delete("/{com_id}/projects/{project_id}")
def delete_project(com_id: str, project_id: str):
    try:
        result = projects_col.delete_one({"com_id": com_id, "project_id": project_id}) ##deleting from mongodb
        s3.delete_project(project_id=project_id) ##delete from S3 
        if result.deleted_count == 0:
            raise HTTPException(404, "Project not found")
        return {"message": "Project deleted successfully"}
    except Exception as e:
        raise HTTPException(500, detail=f"Error deleting project: {e}")


@app.put("/{com_id}/projects/{project_id}")
def update_project(com_id: str, project_id: str, data = Body(...)):
    try:
        result = projects_col.update_one(
            {"com_id": com_id, "project_id": project_id},
            {"$set": {
                "name": data.get("name", ""),
                "description": data.get("description", "")
            }}
        )
        if result.matched_count == 0:
            raise HTTPException(404, "Project not found")
        return {"message": "Project updated successfully"}
    except Exception as e:
        raise HTTPException(500, detail=f"Error updating project: {e}")



@app.post("/ingest/{project_id}")
async def ingest_csv(project_id: str, file: UploadFile = File(...)):
    contents = await file.read()
    try:
        ingestor.ingest_from_object(contents)  # ✅ use existing method
        raw_df = ingestor.get_data()
        # raw_df = reduce_memory_usage(raw_df)
        s3.upload_dataframe(raw_df, project_id)
        projects_col.find_one_and_update(
            {"project_id":project_id},
            {"$set":{
                "data_uploaded": True,
                "total_columns": raw_df.columns.tolist(),
            }}
        )

    except Exception as e:
        raise HTTPException(500, f"Ingest/S3 upload failed: {e}")
    
    return {
        "message": f"Raw data for {project_id} uploaded to S3.",
        "columns": raw_df.columns.tolist()
    }
@app.post("/upload/other_files/{project_id}")
def upload_other_files(project_id: str, files: List[UploadFile] = File(...)):
    uploaded_filenames = []
    subfolder = "other_files"

    for file in files:
        # Read uploaded CSV file
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Convert DataFrame to Parquet format in-memory
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)

        # Replace spaces and change extension to .parquet
        original_name = file.filename.rsplit('.', 1)[0]
        sanitized_name = original_name.replace(" ", "_")
        filename_parquet = f"{sanitized_name}.parquet"

        # Upload to S3
        s3.upload_file(
            file_stream=buffer,
            project_id=project_id,
            filename=filename_parquet,
            subfolder=subfolder
        )

        uploaded_filenames.append(filename_parquet)

    # Update MongoDB with list of uploaded Parquet filenames
    projects_col.update_one(
        {"project_id": project_id},
        {"$set": {"other_files": uploaded_filenames}},
        upsert=True
    )

    return {"uploaded_files": uploaded_filenames}


@app.post("/upload/brand_files/{project_id}")
async def upload_brand_files(project_id: str, files: List[UploadFile] = File(...)):
    uploaded_filenames = []
    subfolder = "brand_files"
    documents =  []
    for file in files:
        contents = file.file.read()
        buffer = io.BytesIO(contents)
        buffer.seek(0)

        # Replace spaces in filename
        sanitized_name = file.filename.replace(" ", "_")

        s3.upload_file(
            file_stream=buffer,
            project_id=project_id,
            filename=sanitized_name,
            subfolder=subfolder
        )
        uploaded_filenames.append(sanitized_name)
        # uploaded_filenames.append(sanitized_name)
        documents.append(docs_processor.ingest_upload_file(upload_file=file))
    chroma_db.add_documents_to_collection(documents=documents,collection_name=rag_config.brand_collection_name)

    # Update MongoDB
    projects_col.update_one(
        {"project_id": project_id},
        {"$set": {"brand_files": uploaded_filenames}},
        upsert=True
    )

    return {"ingested_documents": uploaded_filenames}


@app.post("/upload/domain_files/{project_id}")
async def upload_domain_files(project_id: str, files: List[UploadFile] = File(...)):
    uploaded_filenames = []
    subfolder = "domain_files"
    documents = []
    for file in files:
        contents = file.file.read()
        buffer = io.BytesIO(contents)
        buffer.seek(0)

        # Replace spaces in filename
        sanitized_name = file.filename.replace(" ", "_")

        s3.upload_file(
            file_stream=buffer,
            project_id=project_id,
            filename=sanitized_name,
            subfolder=subfolder
        )
        uploaded_filenames.append(sanitized_name)
        documents.append(docs_processor.ingest_upload_file(upload_file=file))
    chroma_db.add_documents_to_collection(documents=documents,collection_name=rag_config.domain_collection_name)

    # Update MongoDB
    projects_col.update_one(
        {"project_id": project_id},
        {"$set": {"domain_files": uploaded_filenames}},
        upsert=True
    )

    return {"uploaded_files": uploaded_filenames}

# @app.get("/download/{file_category}/{project_id}/{filename}")
# def download_file(file_category: str, project_id: str, filename: str):
#     """
#     file_category: 'brand_files' or 'domain_files'
#     """
#     if file_category not in {"brand_files", "domain_files"}:
#         raise HTTPException(status_code=400, detail="Only 'brand_files' and 'domain_files' are allowed")

#     s3_key = f"{project_id}/{file_category}/{filename}"

#     try:
#         file_data = s3.download_file(project_id=project_id, filename=filename, subfolder=file_category)  # returns bytes
#         buffer = io.BytesIO(file_data)
#         buffer.seek(0)

#         return StreamingResponse(
#             buffer,
#             media_type="application/octet-stream",
#             headers={"Content-Disposition": f"attachment; filename={filename}"}
#         )
#     except Exception as e:
#         raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")

@app.get("/download_all_files/{project_id}")
def download_all_files(project_id: str):
    subfolders = ["brand_files", "domain_files"]
    zip_buffer = io.BytesIO()

    try:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            found_files = False

            for subfolder in subfolders:
                prefix = f"projects/{project_id}/{subfolder}/"
                response = s3.s3.list_objects_v2(Bucket=s3.bucket_name, Prefix=prefix)
                contents = response.get("Contents", [])

                for obj in contents:
                    file_key = obj["Key"]
                    filename_in_zip = file_key.replace(f"projects/{project_id}/", "")
                    file_data = s3.download_file(project_id=project_id, filename=filename_in_zip.split("/")[-1], subfolder=subfolder)
                    zip_file.writestr(filename_in_zip, file_data)
                    found_files = True

        if not found_files:
            raise HTTPException(status_code=404, detail="No brand or domain files found for this project")

        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=docs.zip"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.get("/{com_id}/projects/{project_id}")
def get_project_details(com_id: str = Path(...),
                        project_id: str = Path(...)):
    try:
        result = projects_col.find_one(
            {"com_id": com_id, "project_id": project_id},
            {
                "_id": 0,                    # Exclude MongoDB's default _id
                "com_id": 1,
                "project_id": 1,
                "name": 1,
                "description": 1,
                "created_at": 1,
                "data_uploaded": 1,
                "total_columns": 1,
                "dropped_columns": 1,
                "kpi_columns": 1,
                "important_columns": 1,
                "clusters": 1          # Include if you still want clusters
                # conversations is NOT included
            }
        )
        if not result:
            raise HTTPException(404, detail="Project not found")
        return result

    except Exception as e:
        raise HTTPException(500, detail=f"Error Fetching project: {e}")



@app.get("/projects/{project_id}/get_columns")
def get_columns(project_id:str):
    try:
        df = s3.get_dataframe(project_id=project_id).columns
        return {"columns" : df.to_list()}
    except Exception as e:
        raise HTTPException(status_code=404,detail=str(e))



@app.post("/projects/{project_id}/drop_columns")
def update_dropped_columns(project_id: str, column_list: ColumnList):
    try:
        ingestor.drop_columns(column_list.columns) # changed the argument to column_list.columns by shankersingh01
        result = projects_col.update_one(
            {"project_id": project_id},
            {"$set": {"dropped_columns": column_list.columns}}
        )
        if result.modified_count == 0:
            raise Exception("Project not found or no change made.")
        return {"message": "Dropped columns updated", "dropped_columns": column_list.columns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/{com_id}/projects/")
def create_project(
    com_id: str = Path(...),
    data:NewProject = Body(...)
):
    try:
        project_id = str(uuid.uuid4())
        project_data = {
            "com_id": com_id,  
            "project_id": project_id,
            "name": data.name,
            "description": data.description,
            "created_at": datetime.utcnow().isoformat(),
            "data_uploaded": False,
            "total_columns": [],
            "dropped_columns": [],
            "kpi_columns": [],
            "important_columns": [],
            "clusters": None
            }
        projects_col.insert_one(project_data)
        return {
            "message": "Project created successfully",
            "project_id": project_id,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/projects/{project_id}/kpi_columns")
def update_kpi_columns(project_id: str, column_list: ColumnList):
    try:
        result = projects_col.update_one(
            {"project_id": project_id},
            {"$set": {"kpi_columns": column_list.columns}}
        )
        if result.modified_count == 0:
            raise Exception("Project not found or no change made.")
        return {"message": "KPI columns updated", "kpi_columns": column_list.columns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/projects/{project_id}/important_columns")
def update_important_columns(project_id: str, column_list: ColumnList):
    try:
        result = projects_col.update_one(
            {"project_id": project_id},
            {"$set": {"important_columns": column_list.columns}}
        )
        if result.modified_count == 0:
            raise Exception("Project not found or no change made.")
        return {"message": "Important columns updated", "important_columns": column_list.columns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/projects/{project_id}/reset_columns")
def reset_project_columns(project_id: str):
    try:
        result = projects_col.update_one(
            {"project_id": project_id},
            {
                "$set": {
                    "dropped_columns": [],
                    "kpi_columns": [],
                    "important_columns": [],
                    "total_columns": [],
                }
            }
        )
        if result.matched_count == 0:
            raise HTTPException(404, detail="Project not found")
        return {"message": "All column lists reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Assuming these classes or functions accept the dropped, kpi, and important columns as arguments
@app.post("/process/{project_id}")
async def process_cluster(project_id: str):
    try:
        # Fetching dropped, kpi, and important columns from MongoDB
        project_data = projects_col.find_one({"project_id": project_id})

        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")

        dropped_columns = project_data.get("dropped_columns", [])
        kpi_columns = project_data.get("kpi_columns", [])
        important_columns = project_data.get("important_columns", [])

        # Load raw data from S3
        df_raw = s3.get_dataframe(project_id)
        processor = DataFramePreprocessor(columns_to_drop=dropped_columns)
        # Pass the dropped columns to the processor
        df_proc = processor.fit_transform(df_raw)

        # Pass the kpi and important columns to the engine
        root = engine.build_cluster_trees(raw_df=df_raw,df=df_proc, kpi_columns=kpi_columns, columns_to_analyze=important_columns)

        # Convert the cluster tree to a dictionary format
        clusters = engine.to_dict()
        # Save the processed data to mongodb
        result = projects_col.update_one(
            {"project_id": project_id},
            {"$set": {"clusters": clusters}}
        )

    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}")

    return JSONResponse({"project_id": project_id, "message": clusters})

#Do not use this for now
# @app.post("/projects/{project_id}/get_cluster_tree")
# def get_project_details(project_id:str = Path(...)):
#     try : 
#         result = projects_col.find(
#                 {"project_id": project_id},
#                 {'_id': 0}
#         )
#         keys = list([ row  for row in result])
#         return keys[0]
    
#     except Exception as e :
#         raise HTTPException(500,detail=f"Error Fetching projects: {e}")
    

@app.post("/projects/{project_id}/chat")
async def chat(
    project_id: str = Path(...),
    data: dict = Body(...)
):
    try:
        # Load project from DB
        project = projects_col.find_one({"project_id": project_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Load file & prepare data
        df = s3.get_dataframe(project_id)

        query = data["query"]
        rag_system.initialize(dataframe=df)
        response = rag_system.query(question=query)

        # Save conversation in DB
        new_conversation = {
            "query": query,
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        }

        projects_col.update_one(
            {"project_id": project_id},
            {"$push": {"conversations": new_conversation}}
        )

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred - {e}")


# @app.get("/projects/{project_id}/chat_history")
# async def get_chat_history(
#                         project_id:str = Path(...)
#                      ):
#     try : 
#         if chatManager.hasData != True:
#             return {
#                     "chat_hostory":
#                         chatManager.get_all_responses()
#                     }
#     except Exception as e:
#         return HTTPException(500,f"Error Occured - {e}")


@app.get("/projects/{project_id}/chat_history")
async def chat_history(project_id: str = Path(...)):
    try:
        # Fetch the project by project_id
        project = projects_col.find_one(
            {"project_id": project_id},
            {"_id": 0, "conversations": 1}
        )

        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")

        # Return conversations (empty list if not present)
        conversations = project.get("conversations") or []

        return {"conversations": conversations}

    except HTTPException:
        raise  # Re-raise known HTTP errors
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {e}")


@app.post("/projects/{project_id}/clusters/get_clusters")
async def get_dataframe(project_id:str = Path(...),indexes:List[str] = Body(...)):
    try : 
        data = s3.get_dataframe(project_id)
        indexes=[int(i) for i in indexes]
        data = df.iloc[indexes,:]
        # print(df.to_json())
        return HTMLResponse(data.to_html(),status_code=200)
    
    except Exception as e:
        raise HTTPException(500,f"Error occured - {e}")
    
@app.post("/update-cluster-journey")
async def update_cluster_journey(data: ClusterJourneyUpdate):
    # Check if project exists
    project = projects_col.find_one({"project_id": data.project_id})
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Append new steps to the existing cluster_journey
    updated_journey = project.get("cluster_journey", []) + [step.dict() for step in data.cluster_journey]

    # Update the document
    projects_col.update_one(
        {"project_id": data.project_id},
        {
            "$set": {
                "cluster_journey": updated_journey,
                "cluster_selection_index": data.cluster_selection_index
            }
        }
    )

    return {"message": "Cluster journey updated successfully."}

# ✅ Run App
if __name__ == "__main__":
    app.run()