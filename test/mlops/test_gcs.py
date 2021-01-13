from unittest import TestCase
import os
import shutil

from bavard_ml_common.mlops.gcs import GCSClient


class TestGCSClient(TestCase):
    test_data = {
        "type": "dir",
        "path": "gcs-test",
        "children": [
            {"type": "file", "path": "gcs-test/test-file.txt", "content": "This is a test."},
            {"type": "file", "path": "gcs-test/test-file-2.txt", "content": "This is also a test."}
        ]
    }
    test_bucket = "gcs-client-bucket"

    @classmethod
    def setUpClass(cls) -> None:
        os.makedirs(cls.test_data["path"])
        for child in cls.test_data["children"]:
            with open(child["path"], "w") as f:
                f.write(child["content"])

        cls.client = GCSClient()
        cls.client.create_bucket(cls.test_bucket)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.test_data["path"])

    def test_should_work_with_emulator(self):
        test_file = self.test_data["children"][0]
        # Can upload blob.
        gcs_uri = f"gs://{self.test_bucket}/{test_file['path']}"
        blob = self.client.upload_filename_to_blob(test_file["path"], gcs_uri)
        # Can download blob; contents are correct.
        self.assertEqual(test_file["content"], blob.download_as_text())
        # Can delete blob.
        blob.delete()
