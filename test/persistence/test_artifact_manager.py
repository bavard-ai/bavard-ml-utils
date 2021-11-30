import sys
import time
import typing as t
from unittest import TestCase

import numpy as np

from bavard_ml_utils.persistence.artifact_manager import (
    BaseArtifactManager,
    BaseArtifactRecord,
    BaseDatasetRecord,
    ServiceVersionMetadata,
)
from bavard_ml_utils.persistence.record_store.firestore import FirestoreRecordStore
from test.utils import clear_firestore


class ArtifactRecord(BaseArtifactRecord):
    A: np.ndarray
    b: float


class DatasetRecord(BaseDatasetRecord):
    examples: t.List[str]


class ArtifactManager(BaseArtifactManager):
    def create_artifact_from_dataset(self, dataset: BaseDatasetRecord) -> BaseArtifactRecord:
        # Simulate the artifact being a deterministic output of the dataset's digest and
        # the service version.

        # Source: https://stackoverflow.com/a/18766695
        seed = hash((dataset.digest, self.version)) % ((sys.maxsize + 1) * 2)
        rng = np.random.default_rng(seed)
        return ArtifactRecord(
            agent_id=dataset.agent_id,
            dataset_digest=dataset.digest,
            service_version=self.version,
            updated_at=time.time(),
            A=rng.normal(size=(5, 5)),
            b=rng.random(),
        )


class TestArtifactManager(TestCase):
    def setUp(self):
        clear_firestore()
        self.artifacts: FirestoreRecordStore[ArtifactRecord] = FirestoreRecordStore[ArtifactRecord](
            "artifacts", ArtifactRecord
        )
        self.datasets: FirestoreRecordStore[DatasetRecord] = FirestoreRecordStore[DatasetRecord](
            "datasets", DatasetRecord
        )
        self.versions: FirestoreRecordStore[ServiceVersionMetadata] = FirestoreRecordStore[ServiceVersionMetadata](
            "versions", ServiceVersionMetadata
        )
        self.dataset_records = [
            DatasetRecord(examples=["a", "b", "c"], agent_id="1", updated_at=time.time()),
            DatasetRecord(examples=["d", "e"], agent_id="2", updated_at=time.time()),
        ]

    def test_sync_new_version(self):
        # Simulates the scenario where a new version of the service is being released,
        # and checks to make sure all the proper artifacts are re-indexed for the new version.

        # Setup: create a couple datasets and artifacts for this version.
        mgr = ArtifactManager(self.artifacts, self.datasets, self.versions, "v1")
        for dataset_record in self.dataset_records:
            artifact = mgr.create_artifact_from_dataset(dataset_record)
            mgr.save_artifact(artifact, dataset_record)

        # There should be 2 artifacts saved for the first service version, and none for the second.
        v1_artifacts = list(self.artifacts.get_all(service_version="v1"))
        self.assertEqual(len(v1_artifacts), 2)
        v2_artifacts = list(self.artifacts.get_all(service_version="v2"))
        self.assertEqual(len(v2_artifacts), 0)

        # Act: sync artifacts for a new version of the artifact manager.
        mgr = ArtifactManager(self.artifacts, self.datasets, self.versions, "v2")
        mgr.sync()

        # There should now be two artifacts for this new version, produced from the two datasets added
        # by the previous version.
        v2_artifacts = list(self.artifacts.get_all(service_version="v2"))
        self.assertEqual(len(v2_artifacts), 2)
        # The new service version's artifacts should be for the same dataset versions that the
        # previous version saved.
        self.assertSetEqual(
            {artifact.dataset_digest for artifact in v1_artifacts},
            {artifact.dataset_digest for artifact in v2_artifacts},
        )
        # The data should be the same as well.
        v1_artifacts_by_digest = {artifact.dataset_digest: artifact for artifact in v1_artifacts}
        for v2_artifact in v2_artifacts:
            v1_artifact = v1_artifacts_by_digest[v2_artifact.dataset_digest]
            self.assertEqual(v1_artifact.agent_id, v2_artifact.agent_id)
            self.assertEqual(v1_artifact.dataset_digest, v2_artifact.dataset_digest)
            # The service version and updated_at fields should be different, since they were produced at different
            # times for different versions. The same goes for the data.
            self.assertNotEqual(v1_artifact.updated_at, v2_artifact.updated_at)
            self.assertNotEqual(v1_artifact.service_version, v2_artifact.service_version)
            self.assertNotEqual(v1_artifact.b, v2_artifact.b)
            self.assertFalse((v1_artifact.A == v2_artifact.A).all())

    def test_sync_same_version(self):
        # Simulates the scenario where a new service instance is spun up with the same version of
        # the currently deployed service. For example, when cloud auto-scaling scales up the service.

        # Setup: create a dataset and artifact for this version.
        mgr = ArtifactManager(self.artifacts, self.datasets, self.versions, "v1")
        artifact = mgr.create_artifact_from_dataset(self.dataset_records[0])
        mgr.save_artifact(artifact, self.dataset_records[0])
        artifacts1 = list(self.artifacts.get_all())
        datasets1 = list(self.datasets.get_all())
        self.assertEqual(len(artifacts1), 1)
        self.assertEqual(len(datasets1), 1)
        self.assertEqual(artifacts1[0].service_version, "v1")

        # Act: simuate a new instance being spun up and synced.
        mgr = ArtifactManager(self.artifacts, self.datasets, self.versions, "v1")  # same version
        n_indexed = mgr.sync()

        # `sync` didn't need to do anything.
        self.assertEqual(n_indexed, 0)
        artifacts2 = list(self.artifacts.get_all())
        self.assertEqual(len(artifacts1), 1)
        # The indexed artifact should not have changed.
        artifact1, artifact2 = artifacts1[0], artifacts2[0]
        self.assertEqual(artifact1.service_version, artifact2.service_version)
        self.assertEqual(artifact1.dataset_digest, artifact2.dataset_digest)
        self.assertEqual(artifact1.agent_id, artifact2.agent_id)
        self.assertEqual(artifact1.updated_at, artifact2.updated_at)
        self.assertEqual(artifact1.b, artifact2.b)
        self.assertTrue((artifact1.A == artifact2.A).all())
        # The saved dataset should not have changed either.
        datasets2 = list(self.datasets.get_all())
        self.assertEqual(len(datasets2), 1)
        self.assertEqual(datasets1[0], datasets2[0])

    def test_should_sync_missed_artifacts_on_the_fly(self):
        mgr = ArtifactManager(self.artifacts, self.datasets, self.versions, "v1")
        # Simulate a dataset being saved to a prior service version (no artifact data exists for it for the current
        # service version).
        dataset_record = self.dataset_records[0]
        self.datasets.save(dataset_record)
        # The artifact should not exist in the database for the current service version.
        artifact = self.artifacts.get(ArtifactRecord.make_id("v1", dataset_record.agent_id))
        self.assertIsNone(artifact)
        artifact = mgr.load_artifact(dataset_record.agent_id)  # should compute and save the artifact on the fly
        self.assertIsNotNone(artifact)
        # The artifact should now exist in the database for the current service version, and will no longer
        # need to be computed on the fly.
        artifact = self.artifacts.get(ArtifactRecord.make_id("v1", dataset_record.agent_id))
        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.agent_id, dataset_record.agent_id)
        self.assertEqual(artifact.service_version, "v1")
        self.assertEqual(artifact.dataset_digest, dataset_record.digest)
        self.assertIsNotNone(artifact.A)
        self.assertIsNotNone(artifact.b)

    def test_remove_old_versions(self):
        max_service_versions = 5
        n_old = 2
        version_names = [f"v{i}" for i in range(max_service_versions + n_old)]

        # Create artificial metadata and a couple artifacts for each version.
        for version_name in version_names:
            self.versions.save(ServiceVersionMetadata(name=version_name, synced_at=time.time()))
            mgr = ArtifactManager(self.artifacts, self.datasets, self.versions, version_name)
            for dataset_record in self.dataset_records:
                artifact = mgr.create_artifact_from_dataset(dataset_record)
                mgr.save_artifact(artifact, dataset_record)

        mgr = ArtifactManager(self.artifacts, self.datasets, self.versions, version_names[-1])
        mgr._remove_old_service_versions()

        # Metadata for old versions should have been removed, and data for the newer versions should have been kept.
        versions = list(self.versions.get_all())
        new_version_names = {v.name for v in versions}
        expected_version_names = set(version_names[n_old:])
        self.assertSetEqual(new_version_names, expected_version_names)

        # Artifacts for newer versions should have been preserved.
        for version_name in new_version_names:
            tasks_for_version = list(self.artifacts.get_all(service_version=version_name))
            self.assertSetEqual(
                {task.dataset_digest for task in tasks_for_version}, {rec.digest for rec in self.dataset_records}
            )
            self.assertSetEqual(
                {task.agent_id for task in tasks_for_version}, {rec.agent_id for rec in self.dataset_records}
            )

        # Artifacts for old versions should have been removed.
        for removed_version_name in set(version_names) - new_version_names:
            n_artifacts_for_version = sum(1 for _ in self.artifacts.get_all(service_version=removed_version_name))
            self.assertEqual(n_artifacts_for_version, 0)
