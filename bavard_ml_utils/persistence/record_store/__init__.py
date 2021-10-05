"""
Contains methods for easily persisting `Pydantic <https://pydantic-docs.helpmanual.io/>`_ data structures to and from
arbitrary storage back-ends. Contains a base class for the store behavior, as well as subclasses which allow Google
Cloud Firestore or in-memory to be used as the storage back-end. Supports saving, retrieving, and deleting records by
ID, or performing `WHERE equals` clause-based retrieval and deletion. Pydantic models that include numpy array members
are supported out of the box.
"""
