import tarfile
from tempfile import TemporaryFile
import typing as t

from google.cloud.storage.client import Client
from google.cloud.storage.blob import Blob
from google.auth.credentials import AnonymousCredentials

from bavard_ml_common import config


class GCSClient(Client):
    """Version of `google.cloud.storage.client.Client` that has additional helper methods.
    """

    def __init__(self, **kwargs):
        if kwargs["credentials"] is None and config.IS_TEST:
            # Use anonymous credentials as the default in a test environment.
            # This allows emulators to be used.
            kwargs["credentials"] = AnonymousCredentials()
        super().__init__(**kwargs)

    @staticmethod
    def is_gcs_uri(path: str) -> bool:
        """Returns `True` if `path` is a google cloud storage file path.
        """
        return path.startswith("gs://")

    def to_gcs_tar(self, source_path: str, target_uri: str, arcname: t.Optional[str] = None):
        """Tars the contents of local `source_path` to `target_uri` blob in Google Cloud Storage.
        """
        with TemporaryFile() as temp_file:
            with tarfile.open(fileobj=temp_file, mode="w") as tar:
                tar.add(source_path, arcname=arcname)
            Blob.from_string(target_uri, self).upload_from_file(temp_file)

    def delete_blob(self, uri: str):
        Blob.from_string(uri, self).delete()

    def upload_file_to_blob(self, source_path: str, target_uri: str):
        Blob.from_string(target_uri, self).upload_from_filename(source_path)
