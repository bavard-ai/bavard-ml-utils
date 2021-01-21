import os
import typing as t

from google.cloud.storage.client import Client
from google.cloud.storage.blob import Blob
from google.auth.credentials import AnonymousCredentials


class GCSClient(Client):
    """
    Version of `google.cloud.storage.client.Client` that has additional helper methods, and
    support for GCS emulator.
    """

    def __init__(self, **kwargs):
        if os.getenv("STORAGE_EMULATOR_HOST") is not None:
            # We are in a testing context. Make sure the client's default args
            # work in this emulator scenario.
            if kwargs.get("credentials") is None:
                kwargs["credentials"] = AnonymousCredentials()
            if kwargs.get("project") is None:
                kwargs["project"] = "test"
        super().__init__(**kwargs)

    @staticmethod
    def is_gcs_uri(path: str) -> bool:
        """Returns `True` if `path` is a google cloud storage file path.
        """
        return path.startswith("gs://")

    def delete_blob(self, uri: str):
        Blob.from_string(uri, self).delete()

    def upload_filename_to_blob(self, source_path: str, target_uri: str) -> Blob:
        blob = Blob.from_string(target_uri, self)
        blob.upload_from_filename(source_path)
        return blob

    def download_blob_to_filename(self, uri: str, filename: str):
        blob = Blob.from_string(uri, self)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        blob.download_to_filename(filename)

    def upload_dir(self, source_path: str, target_uri: str):
        """
        Recursively uploads all files in the directory at `source_path`
        to the GCS directory at `target_uri`. Source:
        https://stackoverflow.com/questions/48514933/how-to-copy-a-directory-to-google-cloud-storage-using-google-cloud-python-api
        """
        assert os.path.isdir(source_path)
        for root, _, filenames in os.walk(source_path):
            for filename in filenames:
                file_path_local = os.path.join(root, filename)
                file_path_remote = os.path.join(target_uri, file_path_local[1 + len(source_path):])
                self.upload_filename_to_blob(file_path_local, file_path_remote)

    def download_dir(self, source_uri: str, target_path: str):
        """
        Recursively downloads all files living under `source_uri` in GCS, downloading
        them into the `target_path` directory. Source:
        https://stackoverflow.com/questions/49748910/python-download-entire-directory-from-google-cloud-storage
        """
        bucket_name, bucket_dir = self.parse_gcs_uri(source_uri)
        # Get list of files under `uri` directory
        blobs = self.list_blobs(bucket_name, prefix=bucket_dir)
        for blob in blobs:
            blob_uri = self.get_blob_uri(blob)
            blob_path_local = os.path.join(target_path, blob_uri[1 + len(source_uri):])
            self.download_blob_to_filename(blob_uri, blob_path_local)

    def parse_gcs_uri(self, uri: str) -> t.Tuple[str, str]:
        """Returns the bucket path components of GCS uri `uri`. Raises an error if no path component exists.
        """
        assert self.is_gcs_uri(uri)
        path_components = uri.replace("gs://", "").split("/")
        assert len(path_components) > 1
        bucket_name = path_components[0]
        path = "/".join(path_components[1:])
        return bucket_name, path

    @staticmethod
    def get_blob_uri(blob: Blob) -> str:
        return "gs://" + blob.bucket.name + "/" + blob.name
