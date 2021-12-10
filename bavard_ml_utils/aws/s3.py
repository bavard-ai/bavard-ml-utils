import os

import boto3
from botocore.config import Config


class S3Client:
    def __init__(self):
        endpoint = os.getenv("AWS_ENDPOINT")
        region = os.getenv("AWS_REGION")
        self.resource = boto3.resource("s3", endpoint_url=endpoint, config=Config(region_name=region))
        self.client = boto3.client("s3", endpoint_url=endpoint, config=Config(region_name=region))

    def upload_dir(self, source_path: str, target_bucket: str, target_path: str):
        """
        Recursively uploads all files in the directory at ``source_path`` as objects in the AWS ``target_bucket``, all
        saved under the "directory" ``target_path``.
        """
        assert os.path.isdir(source_path)
        for root, _, filenames in os.walk(source_path):
            for filename in filenames:
                file_path_local = os.path.join(root, filename)
                object_key = os.path.join(target_path, file_path_local[1 + len(source_path) :])
                self.client.upload_file(file_path_local, target_bucket, object_key)

    def download_dir(self, source_bucket: str, source_path: str, target_path: str):
        """
        Recursively downloads all files living under the ``source_path`` directory in the AWS ``source_bucket``.
        Downloads them to the local directory ``target_path``.
        """
        objects = self.resource.Bucket(source_bucket).objects.filter(Prefix=source_path)
        for obj in objects:
            obj_save_path = os.path.join(target_path, obj.key[1 + len(source_path) :])
            save_dir = os.path.abspath(os.path.dirname(obj_save_path))
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            os.makedirs(os.path.dirname(obj_save_path), exist_ok=True)
            self.client.download_file(source_bucket, obj.key, obj_save_path)
