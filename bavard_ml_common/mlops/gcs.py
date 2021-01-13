import os

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
        blob.download_to_filename(filename)
