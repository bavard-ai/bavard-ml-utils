import json
import os
import shutil
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel


def load_json_file(path):
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
