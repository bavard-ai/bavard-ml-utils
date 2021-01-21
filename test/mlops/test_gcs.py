from unittest import TestCase

from bavard_ml_common.mlops.gcs import GCSClient

from test.utils import FileSpec, DirSpec


class TestGCSClient(TestCase):
    test_data_spec = DirSpec(path="gcs-test", children=[
        FileSpec(path="test-file.txt", content="This is a test."),
        FileSpec(path="test-file-2.txt", content="This is also a test."),
        DirSpec(path="subdir", children=[
            FileSpec(path="test-file-3.txt", content="This one too.")
        ])
    ])
    test_bucket_name = "gcs-client-bucket"

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_data_spec.write()
        cls.client = GCSClient()
        cls.client.create_bucket(cls.test_bucket_name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.test_data_spec.remove()

    def tes_can_upload_and_download_blob(self):
        test_file = self.test_data_spec.children[0]
        # Can upload blob.
        gcs_uri = f"gs://{self.test_bucket_name}/{test_file.path}"
        blob = self.client.upload_filename_to_blob(test_file.path, gcs_uri)
        # Can download blob; contents are correct.
        self.assertEqual(test_file.content, blob.download_as_text())
        # Can delete blob.
        blob.delete()

    def test_can_upload_and_download_directory(self):
        gcs_upload_dir = f"gs://{self.test_bucket_name}/temp-data"
        # Upload directory (including a subdirectory).
        self.client.upload_dir(self.test_data_spec.path, gcs_upload_dir)
        # Download directory.
        self.client.download_dir(gcs_upload_dir, "gcs-test-copy")

        # Folder that was uploaded and downloaded should recursively have
        # the same contents as the original one.
        downloaded_spec = DirSpec.from_path("gcs-test-copy")
        for child, dchild in zip(
            sorted(self.test_data_spec.children, key=lambda c: c.path),
            sorted(downloaded_spec.children, key=lambda c: c.path)
        ):
            self.assertEqual(child, dchild)

        # Clean up.
        downloaded_spec.remove()
