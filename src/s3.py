import boto3
import pandas as pd
from boto3 import resource
from botocore.exceptions import ClientError
from typing import List, Optional
import io



class S3Client:
    def __init__(self, bucket_name: str, access_key:Optional[str] = None,secret_key:Optional[str] = None,region_name: Optional[str] = None):
        self.resource = resource("s3", aws_access_key_id=access_key,aws_secret_access_key=secret_key,region_name=region_name)
        self.bucket = self.resource.Bucket(bucket_name)
        self.bucket_name = bucket_name
        if secret_key and access_key is None:
            raise ValueError("Access key must be provided if secret key is provided.")
        self.__secret_key = secret_key
        self.__access_key = access_key
        self.s3 = boto3.client("s3", aws_access_key_id = access_key,aws_secret_access_key = secret_key,region_name=region_name)

    def create_folder(self, folder_name: str) -> str:
        try:
            self.s3.put_object(Bucket=self.bucket_name, Key=(folder_name + "/"))
            return f"Folder {folder_name} created successfully."
        except ClientError as e:
            raise Exception(f"Folder creation failed: {str(e)}")
        

    def upload_file(self, file_stream,folder_name,filename: str) -> str:
        try:
            root_folder = "projects/"
            if folder_name and not folder_name.endswith("/"):
                folder_name += "/"
                folder_name = root_folder + folder_name
                self.s3.upload_fileobj(file_stream, self.bucket_name, folder_name + filename)
            else:
                self.s3.upload_fileobj(file_stream, self.bucket_name, filename)
            return f"Uploaded {filename} successfully."
        except ClientError as e:
            raise Exception(f"Upload failed: {str(e)}")

    def download_file(self, filename: str) -> bytes:
        try:
            s3_object = self.s3.get_object(Bucket=self.bucket_name, Key=filename)
            return s3_object["Body"].read()
        except ClientError as e:
            raise Exception(f"Download failed: {str(e)}")
        
    def getFile(self, project_id:str) -> io.BytesIO:
        try:
            folder = "projects/"
            # file_path = folder + project_id + "/"
            s3_object = self.s3.get_object(Bucket=self.bucket_name, Key=folder + project_id + "/raw_data.parquet")
            return io.BytesIO(s3_object["Body"].read())
        except ClientError as e:
            raise Exception(f"Download failed: {str(e)}")

    def list_projects(self) -> List[str]:
        try:
            response = self.bucket.objects.all()
            return [str(content.key).split('/')[1] for content in response]
        except ClientError as e:
            raise Exception(f"List failed: {str(e)}")


    def upload_dataframe(self,df:pd.DataFrame,project_id:str):
        buffer = io.BytesIO()
        file_name = "raw_data.parquet"
        df.to_parquet(buffer)
        buffer.seek(0)
        self.upload_file(buffer,folder_name=project_id,filename = file_name)
        
    
    # def get_dataframe(self,project_id:str):
    #     df = pd.read_parquet(self.getFile(project_id))

    def update_file(self, file_stream, filename: str) -> str:
        # S3 doesn't distinguish between create and update – both use upload
        return self.upload_file(file_stream, filename)

    def delete_pro(self, filename: str) -> str:
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=filename)
            return f"Deleted {filename} successfully."
        except ClientError as e:
            raise Exception(f"Delete failed: {str(e)}")
