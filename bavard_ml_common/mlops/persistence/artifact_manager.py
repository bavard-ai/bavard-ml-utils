import time
import typing as t
from abc import ABC, abstractmethod

from fastapi import HTTPException, status
from loguru import logger

from bavard_ml_common.mlops.persistence.record_store.base import BaseRecordStore, Record
from bavard_ml_common.types.utils import hash_model


class ServiceVersionMetadata(Record):
    name: str
    synced_at: float  # time the service version was most recently synced.

    def get_id(self) -> str:
        return self.name


class BaseDatasetRecord(Record):
    """
    Base class for a generic dataset object. Supports versioning of the dataset (via its `digest` attribute). Used to
    create `ArtifactRecord`s from. Override with additional attributes to store the actual dataset data in this object.
    """

    agent_id: str
    updated_at: float
    digest: t.Optional[str]  # hash of this record.

    def __init__(self, **data):
        """Custom constructor. Includes a post-init step to update the record's digest."""
        super().__init__(**data)
        self.digest = self.compute_digest()

    def compute_digest(self):
        """
        Creates a deterministic hash of the dataset. This is needed so we know which dataset version(s) a service
        already has artifacts computed for. Without a hash, we couldn't easily know which version of a dataset an
        artifact was computed for.
        """
        return hash_model(self, exclude={"updated_at", "digest"})

    def get_id(self) -> str:
        return self.agent_id


class BaseArtifactRecord(Record):
    """
    Base class for an artifact produced by a versioned dataset and a versioned model. Override with additional
    attributes to store the actual artifact data in this object.
    """

    agent_id: str
    dataset_digest: str  # the hash of the dataset that was used to produce this artifact
    service_version: str  # the version of the model that produced this artifact
    updated_at: float

    def get_id(self) -> str:
        return self.make_id(self.service_version, self.agent_id)

    @staticmethod
    def make_id(service_version: str, agent_id: str):
        """
        We have an artifact's database ID be the concatenation of the service version that created it, and the
        agent that the artifact is for. We do this because for a given service version, we only need to persist one
        artifact version per agent at a time.
        """
        return f"{service_version}-{agent_id}"


class BaseArtifactManager(ABC):
    """
    Abstract base class which provides a simple API for creating, persisting, and managing artifacts produced by a
    versioned machine learning (ML) model. Each artifact is associated with an agent. An artifact is the deterministic
    product of a specific ML model version, and a specific version of a specific dataset.
    """

    def __init__(
        self,
        artifacts: BaseRecordStore[BaseArtifactRecord],
        datasets: BaseRecordStore[BaseDatasetRecord],
        versions: BaseRecordStore[ServiceVersionMetadata],
        version: str,
        *,
        max_service_versions=5,
    ):
        self.version = version
        self._max_service_versions = max_service_versions
        self._artifacts = artifacts
        self._datasets = datasets
        self._versions = versions

    @abstractmethod
    def create_artifact_from_dataset(self, dataset: BaseDatasetRecord) -> BaseArtifactRecord:
        """Takes `dataset`, and the model associated with version `self.version`, and produces an artifact."""
        pass

    def delete_artifact(self, agent_id: str):
        """Deletes from Firestore a dataset and all artifacts associated with it."""
        dataset_record = self._datasets.get(agent_id)
        if dataset_record is None:
            self.raise_no_dataset(agent_id)
        self._datasets.delete(agent_id)
        self._artifacts.delete_all(agent_id=agent_id)  # get rid of any old versions as well

    def save_artifact(self, artifact: BaseArtifactRecord, dataset: BaseDatasetRecord):
        """
        Serializes and saves the artifact. Also saves the dataset that was used to create the artifact, so this
        manager can recreate the artifact at any time if needed.
        """
        self._artifacts.save(artifact)
        self._datasets.save(dataset)

    def load_artifact(self, agent_id: str) -> BaseArtifactRecord:
        """
        Load an agent's artifact from the database. If the agent's dataset exists but its artifact doesn't, then compute
        it, save it (so its available next time), and then return it.
        """
        dataset = self._datasets.get(agent_id)
        if dataset is None:
            self.raise_no_dataset(agent_id)
        artifact = self._artifacts.get(BaseArtifactRecord.make_id(self.version, agent_id))
        if artifact is not None and artifact.dataset_digest == dataset.digest:
            return artifact
        else:
            logger.info(
                f"artifact for agent {agent_id} does not exist for service version {self.version} and dataset "
                f"digest {dataset.digest}; creating it now"
            )
            # Artifact has not yet been created for this dataset version and service version.
            artifact = self.create_artifact_from_dataset(dataset)
            self.save_artifact(artifact, dataset)
            return artifact

    def sync(self) -> int:
        """
        Ensures this service version has its own artifact for all currently saved datasets. Returns the number of
        artifacts that had to be created for that to happen.
        """
        digest2agent = {dataset.digest: dataset.agent_id for dataset in self._datasets.get_all()}
        all_dataset_digests = set(digest2agent.keys())
        datasets_currently_indexed = {
            artifact.dataset_digest for artifact in self._artifacts.get_all(service_version=self.version)
        }
        datasets_to_index = all_dataset_digests - datasets_currently_indexed
        logger.info(
            f"creating artifact for {len(datasets_to_index)}/{len(all_dataset_digests)} "
            f"existing datasets for service version {self.version}"
        )
        for digest in datasets_to_index:
            dataset = self._datasets.get(digest2agent[digest])
            logger.info(
                f"creating artifact for dataset digest={dataset.digest} associated with agent {digest2agent[digest]}"
            )
            artifact = self.create_artifact_from_dataset(dataset)
            self.save_artifact(artifact, dataset)

        self._versions.save(ServiceVersionMetadata(name=self.version, synced_at=time.time()))
        logger.info("service version sync utility finished successfully.")
        self._remove_old_service_versions()
        return len(datasets_to_index)

    @staticmethod
    def raise_no_dataset(agent_id: str):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"no dataset exists for agent id {agent_id}; artifact cannot be retrieved or computed",
        )

    def _remove_old_service_versions(self):
        """
        Removes all artifacts for any old service versions. We only keep data for the `self._max_service_versions` most
        recent service versions. The old versions are the ones that have been synced least recently.
        """
        versions = list(self._versions.get_all())
        n_to_remove = len(versions) - self._max_service_versions
        if n_to_remove > 0:
            logger.info(f"removing data for {n_to_remove} old service versions")
            # Remove the oldest versions.
            versions_to_remove = sorted(versions, key=lambda v: v.synced_at)[:n_to_remove]
            for version in versions_to_remove:
                self._artifacts.delete_all(service_version=version.name)
                self._versions.delete(version.get_id())
