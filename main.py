import json
import smtplib
import random
import uuid
from uuid import uuid4
import bcrypt
from fastapi import FastAPI, HTTPException, Query, Path, Body, UploadFile, File,Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import redis
from typing import List, Optional
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
import os
import string
import io
from io import BytesIO
import pandas as pd
from src.s3 import S3Client
from src.components.dataingestion import DataIngestion
from src.components.datapreprocessing import DataFramePreprocessor
from src.components.clustering import ClusteringEngine
from src.chat import ChatEngine


import requests 



load_dotenv()
app = FastAPI()

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
processor = DataFramePreprocessor()
engine    = ClusteringEngine()
chatManager = ChatEngine(os.getenv("PANDASAI_KEY"))

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
            {"name": 1, "description": 1,"project_id":1 ,"_id": 0}
        )

        projects = [{"name": row.get("name", ""), 
                     "description": row.get("description", ""),
                     "project_id": row.get("project_id")
                     } for row in rows]

        return {"projects": projects}
    except Exception as e:
        raise HTTPException(500, detail=f"Error fetching projects: {e}")

# ✅ Get Projects
# Added project_id to the response by shankersingh01
@app.get("/{com_id}/projects/")
def get_projects(com_id: str = Path(...)):
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
        raise HTTPException(500, detail=f"Error fetching projects: {e}"
    

@app.get("/{com_id}/projects/{project_id}")
def get_project_details(com_id:str = Path(...),
                        project_id:str = Path(...)):
    try : 
        result = projects_col.find(
                {"com_id": com_id,
                "project_id": project_id,
                },
                {'_id': 0}
        )
        keys = list([ row  for row in result])
        return keys[0]
    
    except Exception as e :
        raise HTTPException(500,detail=f"Error Fetching projects: {e}")

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
        s3.upload_dataframe(raw_df, project_id)
        projects_col.find_one_and_update(
            {"project_id":project_id},
            {"$set":{
                "data_uploaded":"true"
            }}
        )

    except Exception as e:
        raise HTTPException(500, f"Ingest/S3 upload failed: {e}")
    
    return {
        "message": f"Raw data for {project_id} uploaded to S3.",
        "columns": raw_df.columns.tolist()
    }

@app.get("/projects/{project_id}/get_data")
def get_data(project_id:str):
    try : 
        data = s3.getFile(project_id)
        print(data)
        df = pd.read_parquet(data,)
        print(df.to_json())
        return Response(df.to_json(orient="records"), media_type="application/json")
    
    except Exception as e:
        raise HTTPException(500,f"Error occured - {e}")
    


@app.get("/projects/{project_id}/get_columns")
def get_columns(project_id:str):
    try:
        df = pd.read_parquet(s3.getFile(project_id=project_id)).columns
        return {"columns" : df.to_list()}
    except Exception as e:
        raise HTTPException(status_code=404,detail=str(e))



@app.post("/projects/{project_id}/drop_columns")
def update_dropped_columns(project_id: str, column_list: ColumnList):
    try:
        ingestor.drop_columns(column_list)
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
                    "important_columns": []
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
        raw_stream = s3.getFile(project_id)
        df_raw = pd.read_parquet(raw_stream)

        # Pass the dropped columns to the processor
        df_proc = processor.fit_transform(df_raw, dropped_columns=dropped_columns)

        # Pass the kpi and important columns to the engine
        root = engine.build_cluster_tree(df_proc, kpi_columns=kpi_columns, columns_to_analyze=important_columns)

        # Convert the cluster tree to a dictionary format
        clusters = engine.to_dict()
        # Save the processed data to mongodb
        result = projects_col.update_one(
            {"project_id": project_id},
            {"$set": {"clusters": clusters}}
        )

    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}")

    return JSONResponse({"project_id": project_id, "cluster_tree": clusters})


@app.post("/projects/{project_id}/chat")
async def chat(
    project_id = Path(...),
    data = Body(...)
):
    try : 
        chatManager.read_data(project_id=project_id)
        response = chatManager.chat_with_data(data["query"])  ## Can we implement persistance for this ??? maybe a mongodb field where it says chat_history?
        return response
    except Exception as e:
        return HTTPException(500,f"Error occured - {e}")

@app.get("/projects/{project_id}/chat_history")
def get_chat_history(
                        project_id:str = Path(...)
                     ):
    try : 
        if chatManager.hasData != True:
            return {
                    "chat_hostory":
                        chatManager.get_all_responses()
                    }
    except Exception as e:
        return HTTPException(500,f"Error Occured - {e}")


# ✅ Run App
if __name__ == "__main__":
    app.run()