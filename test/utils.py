import json
import os
import shutil
import typing as t
from abc import ABC, abstractmethod

import boto3
import requests
from botocore.config import Config
from pydantic import BaseModel

from test.config import FIRESTORE_EMULATOR_HOST


def load_json_file(path) -> t.Any:
    with open(path) as f:
        return json.load(f)


class FileSystemObjectSpec(ABC):
    @abstractmethod
    def write(self, parent: str = ""):
        pass


class FileSpec(BaseModel, FileSystemObjectSpec):
    path: str  # should be relative to parent, unless there is no parent
    content: str

    def write(self, parent: str = ""):
        with open(os.path.join(parent, self.path), "w") as f:
            f.write(self.content)

    @classmethod
    def from_path(cls, path: str, parent: str = "") -> "FileSpec":
        """
        `parent` will be excluded from this instance's path attribute,
        keeping the path attribute relative.
        """
        with open(os.path.join(parent, path)) as f:
            content = f.read()
        return cls(path=path, content=content)


class DirSpec(BaseModel, FileSystemObjectSpec):
    """Recursively write this directory and all of its children."""

    path: str  # should be relative to parent, unless there is no parent
    children: t.List[FileSystemObjectSpec]

    class Config:
        arbitrary_types_allowed = True

    def write(self, parent: str = ""):
        full_path = os.path.join(parent, self.path)
        os.makedirs(full_path)
        for child in self.children:
            child.write(full_path)

    def remove(self, parent: str = ""):
        shutil.rmtree(os.path.join(parent, self.path))

    @classmethod
    def from_path(cls, path: str, parent: str = "") -> "DirSpec":
        """
        Build out a `DirSpec` instance for the directory at
        `os.path.join(parent, path)`, along with all of its children.
        """
        spec = cls(path=path, children=[])
        base_path = os.path.join(parent, path)
        for child_name in os.listdir(base_path):
            child_path = os.path.join(base_path, child_name)
            if os.path.isdir(child_path):
                spec.children.append(cls.from_path(child_name, base_path))
            elif os.path.isfile(child_path):
                spec.children.append(FileSpec.from_path(child_name, base_path))
        return spec


def clear_firestore():
    # Clear the test database.
    # Source: https://firebase.google.com/docs/emulator-suite/connect_firestore#clear_your_database_between_tests
    res = requests.delete(f"http://{FIRESTORE_EMULATOR_HOST}/emulator/v1/projects/test/databases/(default)/documents")
    res.raise_for_status()


def create_dynamodb_table(table_name: str, *, pk_field="id", sort_key_field=None, sort_key_type=None):
    dynamodb = boto3.resource(
        "dynamodb", endpoint_url=os.getenv("AWS_ENDPOINT"), config=Config(region_name=os.getenv("AWS_REGION"))
    )
    # Source:
    valid_ddb_data_types = ["S", "N", "B", "BOOL"]
    if sort_key_field is None:
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[{"AttributeName": pk_field, "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": pk_field, "AttributeType": "S"}],
            ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
        )
    if sort_key_field is not None:
        if sort_key_type is None or sort_key_type not in valid_ddb_data_types:
            # Source: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/DynamoDBMapper.DataTypes.html
            raise Exception("sort key data type is either not provided or it is a wrong type")
        else:
            table = dynamodb.create_table(
                TableName=table_name,
                KeySchema=[
                    {"AttributeName": pk_field, "KeyType": "HASH"},
                    {"AttributeName": sort_key_field, "KeyType": "RANGE"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": pk_field, "AttributeType": "S"},
                    {"AttributeName": sort_key_field, "AttributeType": sort_key_type},
                ],
                ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
            )
    table.meta.client.get_waiter("table_exists").wait(TableName=table_name)
    return table
