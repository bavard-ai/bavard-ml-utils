import json
from unittest import TestCase
from uuid import uuid4

from bavard_ml_utils.gcp.logging import Severity, log


class TestLogger:
    def __init__(self):
        self.call_count = 0
        self.logged = None

    def __call__(self, body: str):
        self.call_count += 1
        self.logged = body

    def reset(self):
        self.call_count = 0
        self.logged = None


class TestLogging(TestCase):
    def setUp(self):
        self.trace_id = uuid4()
        # Has the format `TRACE_ID/SPAN_ID;o=TRACE_TRUE`. Source: https://cloud.google.com/trace/docs/setup#force-trace
        self.trace_header = f"{self.trace_id}/1;o=1"
        self.gcp_project_id = "test"
        self.logger = TestLogger()

    def test_can_log(self):
        self.assertFalse(log())  # logging nothing should be a noop, and return false
        self.assertTrue(log(message="testing"))

    def test_log_severity(self):
        with self.assertRaises(ValueError):
            log(severity="foo")  # invalid severity
        self.assertTrue(log(severity="DEBUG", message="hello", logger=self.logger))  # can accept strings
        self.assertTrue(log("ERROR", message="world", logger=self.logger))  # argument name is optional
        self.assertTrue(log(severity=Severity.INFO, message="hello", logger=self.logger))  # or enum values
        self.assertEqual(self.logger.call_count, 3)
        # The message should be present in the log entry body.
        self.assertEqual(json.loads(self.logger.logged)["message"], "hello")

    def test_links_with_trace(self):
        self.assertTrue(
            log(trace_header=self.trace_header, gcp_project_id=self.gcp_project_id, foo="bar", logger=self.logger)
        )
        self.assertEqual(self.logger.call_count, 1)
        self.assertIsNotNone(self.logger.logged)
        data = json.loads(self.logger.logged)
        self.assertEqual(data["logging.googleapis.com/trace"], f"projects/{self.gcp_project_id}/traces/{self.trace_id}")
