import inspect
import typing as t

import requests
from fastapi import HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from bavard_ml_utils.utils import ImportExtraError


try:
    from google.cloud import error_reporting
except ImportError:
    _has_gcp_deps = False
else:
    _has_gcp_deps = True
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException


error_types = {"ERROR", "MINOR", "MAJOR", "CRITICAL"}


def report_error(
    message: str,
    slack_url: t.Optional[str],
    gcp_project_id: t.Optional[str],
    severity: str = "ERROR",
):
    """Reports an error ``message`` on Slack || Google Cloud || Both.

    Parameters
    ----------
    message : str
        The error message to send.
    slack_url : str, optional
        An optional Slack webhook URL. If provided, ``message`` will be sent to this URL, prepended by the ``:boom:``
        icon.
    gcp_project_id : str, optional
        If provided, this message will be sent to google cloud error reporting, associated with ``gcp_project_id``.
        Requires the ``gcp`` extra for this package to be installed.
    """
    assert severity in error_types
    error = f":boom: {severity}: {message}"

    if slack_url is not None:
        try:
            requests.post(slack_url, json={"text": error})  # report on slack
        except Exception:
            logger.exception("encountered error while reporting error to Slack")

    if gcp_project_id is not None:
        if not _has_gcp_deps:
            function_name = inspect.stack()[0][3]  # source: https://stackoverflow.com/a/5067654
            raise ImportExtraError("gcp", function_name)
        try:
            error_reporting.Client(gcp_project_id).report(error)  # report on GCP
        except Exception:
            logger.exception("encountered error while reporting error to GCP Error Reporting")


def make_error_reporting_route_handler_class(
    *,
    msg_prefix: str = "",
    slack_webhook_url: t.Optional[str] = None,
    google_cloud_project: t.Optional[str] = None,
) -> t.Type[APIRoute]:
    """
    Makes a FastAPI route handler class which intercepts, logs, and reports all internal errors that occur in a
    FastAPI app. Intercepts all ``Exception`` objects which are uncaught by the app, as well as all 500-level errors
    intentially thrown by your app.

    Example usage:

    >>> from fastapi import FastAPI
    ...
    ... app = FastAPI()
    ... app.router.route_class = make_error_reporting_route_handler_class(
    ...     msg_prefix="SERVICE 123: "
    ... )
    ...
    ... # Then define all routes on `app` like normal.

    Related docs: https://fastapi.tiangolo.com/advanced/custom-request-and-route/

    Parameters
    ----------
    msg_prefix : str, optional
        If provided, this string will be pre-pended to the error messages of all intercepted errors before they are
        logged and reported.
    slack_webhook_url : str, optional
        If provided, error messages for all intercepted errors will be posted to this Slack url.
    google_cloud_project : str, optional
        If provided, error messages for all intercepted errors will be reported to GCP Error Reporting for the
        ``google_cloud_project`` GCP project name.
    """

    class ErrorReportingRouteHandler(APIRoute):
        def get_route_handler(self) -> t.Callable:
            original_route_handler = super().get_route_handler()

            async def custom_route_handler(request: Request) -> Response:
                try:
                    # Try to handle the route as normal.
                    return await original_route_handler(request)
                except Exception as exc:
                    # Some type of exception has occurred. If it's a normal Python exception or a 500-level
                    # exception, log it here and notify the team.
                    error_to_report = None
                    is_fastapi_error = isinstance(exc, (StarletteHTTPException, RequestValidationError))
                    if not is_fastapi_error:
                        # This is an unknown internal exception.
                        error_to_report = msg_prefix + str(exc)

                    if isinstance(exc, StarletteHTTPException) and exc.status_code >= 500:
                        # This is a 500-level internal exception.
                        error_to_report = msg_prefix + exc.detail

                    if error_to_report is not None:
                        # Log and report the error.
                        logger.exception(error_to_report)
                        report_error(error_to_report, slack_webhook_url, google_cloud_project)

                    if is_fastapi_error:
                        raise exc
                    else:
                        # Turn the normal Python exception into an HTTP exception with a message for the caller.
                        was_team_notified = slack_webhook_url is not None and error_to_report is not None
                        server_response = "An internal error has occurred."
                        if was_team_notified:
                            server_response += " Out team has been notified of this incident."
                        raise HTTPException(500, server_response)

            return custom_route_handler

    return ErrorReportingRouteHandler
