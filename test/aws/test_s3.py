from unittest import TestCase

from bavard_ml_utils.aws.s3 import S3Client
from test.utils import DirSpec, FileSpec


class TestS3Client(TestCase):
    test_data_spec = DirSpec(
        path="aws-test",
        children=[
            FileSpec(path="test-file.txt", content="This is a test."),
            FileSpec(path="test-file-2.txt", content="This is also a test."),
            DirSpec(path="subdir", children=[FileSpec(path="test-file-3.txt", content="This one too.")]),
        ],
    )
    test_bucket_name = "aws-client-bucket"

    @classmethod
    def setUpClass(cls):
        cls.test_data_spec.write()
        cls.s3 = S3Client()
        cls.s3.resource.Bucket(cls.test_bucket_name).create()

    @classmethod
    def tearDownClass(cls):
        cls.test_data_spec.remove()

    def tes_can_upload_and_download_object(self):
        test_file = self.test_data_spec.children[0]
        # Can upload object.
        obj = self.s3.resource.Object(self.test_bucket_name, test_file.path)
        obj.upload_file(test_file.path)
        # Can download object; contents are correct.
        self.assertEqual(test_file.content, obj.get()["Body"].read().decode("utf-8"))
        # Can delete object.
        obj.delete()

    def test_can_upload_and_download_directory(self):
        # Upload directory (including a subdirectory).
        self.s3.upload_dir(self.test_data_spec.path, self.test_bucket_name, "temp-data")
        # Download directory.
        self.s3.download_dir(self.test_bucket_name, "temp-data", "aws-test-copy")

        # Folder that was uploaded and downloaded should recursively have
        # the same contents as the original one.
        downloaded_spec = DirSpec.from_path("aws-test-copy")
        for child, dchild in zip(
            sorted(self.test_data_spec.children, key=lambda c: c.path),
            sorted(downloaded_spec.children, key=lambda c: c.path),
        ):
            self.assertEqual(child, dchild)

        # Clean up.
        downloaded_spec.remove()
