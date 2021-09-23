import hashlib

from pydantic import BaseModel


def hash_model(obj: BaseModel, **kwargs) -> str:
    """Creates a deterministic hash of `obj`, which is some pydantic model."""
    digest = hashlib.sha256(obj.json(sort_keys=True, **kwargs).encode("utf-8")).hexdigest()
    return digest
