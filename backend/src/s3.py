import boto3
import pandas as pd
from boto3 import resource
from botocore.exceptions import ClientError
from typing import List, Optional
import io


class S3Client:
    def __init__(self, bucket_name: str, access_key: Optional[str] = None,
                 secret_key: Optional[str] = None, region_name: Optional[str] = None):
        if secret_key and not access_key:
            raise ValueError("Access key must be provided if secret key is provided.")

        self.bucket_name = bucket_name
        self.__secret_key = secret_key
        self.__access_key = access_key

        self.resource = resource("s3",
                                 aws_access_key_id=access_key,
                                 aws_secret_access_key=secret_key,
                                 region_name=region_name)
        self.bucket = self.resource.Bucket(bucket_name)

        self.s3 = boto3.client("s3",
                               aws_access_key_id=access_key,
                               aws_secret_access_key=secret_key,
                               region_name=region_name)

    def create_folder(self, folder_name: str) -> str:
        try:
            self.s3.put_object(Bucket=self.bucket_name, Key=(folder_name.rstrip("/") + "/"))
            return f"Folder {folder_name} created successfully."
        except ClientError as e:
            raise Exception(f"Folder creation failed: {str(e)}")

    def upload_file(self, file_stream, project_id: str, filename: str, subfolder: Optional[str] = "") -> str:
        try:
            root_folder = f"projects/{project_id}/"
            if subfolder and not subfolder.endswith("/"):
                subfolder += "/"
            key = f"{root_folder}{subfolder}{filename}"
            self.s3.upload_fileobj(file_stream, self.bucket_name, key)
            return f"Uploaded {filename} to {key} successfully."
        except ClientError as e:
            raise Exception(f"Upload failed: {str(e)}")

    def download_file(self, project_id: str, filename: str, subfolder: Optional[str] = "") -> bytes:
        try:
            folder_path = f"projects/{project_id}/"
            if subfolder and not subfolder.endswith("/"):
                subfolder += "/"
            key = f"{folder_path}{subfolder}{filename}"
            s3_object = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            return s3_object["Body"].read()
        except ClientError as e:
            raise Exception(f"Download failed: {str(e)}")

    def get_file_as_buffer(self, project_id: str, filename: str, subfolder: Optional[str] = "") -> io.BytesIO:
        return io.BytesIO(self.download_file(project_id, filename, subfolder))

    def upload_dataframe(self, df: pd.DataFrame, project_id: str, subfolder: Optional[str] = ""):
        buffer = io.BytesIO()
        df.to_parquet(buffer)
        buffer.seek(0)
        self.upload_file(buffer, project_id, "raw_data.parquet", subfolder)

    def get_dataframe(self, project_id: str, subfolder: Optional[str] = "") -> pd.DataFrame:
        buffer = self.get_file_as_buffer(project_id, "raw_data.parquet", subfolder)
        return pd.read_parquet(buffer)

    def update_file(self, file_stream, project_id: str, filename: str, subfolder: Optional[str] = "") -> str:
        return self.upload_file(file_stream, project_id, filename, subfolder)

    def delete_file(self, project_id: str, filename: str, subfolder: Optional[str] = "") -> str:
        try:
            folder_path = f"projects/{project_id}/"
            if subfolder and not subfolder.endswith("/"):
                subfolder += "/"
            key = f"{folder_path}{subfolder}{filename}"
            self.s3.delete_object(Bucket=self.bucket_name, Key=key)
            return f"Deleted {filename} from {key} successfully."
        except ClientError as e:
            raise Exception(f"Delete failed: {str(e)}")

    def list_projects(self) -> List[str]:
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix="projects/")
            if "Contents" not in response:
                return []
            project_ids = set()
            for obj in response["Contents"]:
                parts = obj["Key"].split("/")
                if len(parts) > 1:
                    project_ids.add(parts[1])
            return list(project_ids)
        except ClientError as e:
            raise Exception(f"List projects failed: {str(e)}")

    def list_project_files(self, project_id: str, subfolder: Optional[str] = "") -> List[str]:
        try:
            prefix = f"projects/{project_id}/"
            if subfolder and not subfolder.endswith("/"):
                subfolder += "/"
            prefix += subfolder
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            if "Contents" not in response:
                return []
            return [item["Key"] for item in response["Contents"] if not item["Key"].endswith("/")]
        except ClientError as e:
            raise Exception(f"List project files failed: {str(e)}")
