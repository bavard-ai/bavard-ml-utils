import typing as t
import requests
from google.cloud import error_reporting
from loguru import logger


error_types = {'ERROR', 'MINOR', 'MAJOR', 'CRITICAL'}


def report_error(
    message: str,
    slack_url: t.Optional[str],
    gcp_project_id: t.Optional[str],
    severity: str = 'ERROR',
):
    """Reports errors on Slack || Google Cloud || Both
    """
    assert severity in error_types
    error = f":boom: {severity}: {message}"

    if slack_url is not None:
        try:
            requests.post(slack_url, json={"text": error})  # report on slack
        except Exception:
            logger.exception("encountered error while reporting error to Slack")

    if gcp_project_id is not None:
        try:
            error_reporting.Client(gcp_project_id).report(error)  # report on GCP
        except Exception:
            logger.exception("encountered error while reporting error to GCP Error Reporting")
