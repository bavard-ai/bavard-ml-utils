import json
import typing as t
from enum import Enum


class Severity(Enum):
    """
    GCP Logging log severity levels
    (`Source <https://cloud.google.com/logging/docs/reference/v2/rest/v2/LogEntry#LogSeverity>`_).
    """

    DEFAULT = "DEFAULT"
    """The log entry has no assigned severity level."""

    DEBUG = "DEBUG"
    """Debug or trace information."""

    INFO = "INFO"
    """Routine information, such as ongoing status or performance."""

    NOTICE = "NOTICE"
    """Normal but significant events, such as start up, shut down, or a configuration change."""

    WARNING = "WARNING"
    """Warning events might cause problems."""

    ERROR = "ERROR"
    """Error events are likely to cause problems."""

    CRITICAL = "CRITICAL"
    """Critical events cause more severe problems or outages."""

    ALERT = "ALERT"
    """A person must take an action immediately."""

    EMERGENCY = "EMERGENCY"
    """One or more systems are unusable."""


def log(
    severity: t.Union[Severity, str] = Severity.DEFAULT,
    *,
    trace_header: t.Optional[str] = None,
    gcp_project_id: t.Optional[str] = None,
    logger: t.Callable[[str], t.Any] = print,
    labels: t.Optional[t.Dict[str, str]] = None,
    **fields,
):
    """
    Logs ``**fields`` out as a JSON object string. GCP Cloud Run automatically parses the fields, and also captures
    any special fields, like ``severity`` and ``message``.

    Parameters
    ----------
    severity : str or Severity, optional
        The severity of the log entry. Defaults to :attr:`Severity.DEFAULT`, which is no severity.
    trace_header : str, optional
        If this log entry is being logged in the context of an HTTP request and response occuring in GCP Cloud Run, then
        the request will have a ``X-Cloud-Trace-Context`` header. If the value of that header is passed to this function
        as the ``trace_header`` field, this log entry will be associated with that trace, and the entry's fields will be
        nested under the trace's log entry. *Note*: the ``gcp_project_id`` field must also be passed in.
    gcp_project_id : str, optional
        If applicable, the ID of the GCP project the code is running under. Required for this log entry to be associated
        with a trace.
    logger : callable, optional
        The function that will do the logging to the console. Default is the builtin ``print``, which is sufficient for
        GCP to capture the log entry.
    labels : dict, optional
        If provided, will be used as official GCP logging labels for this log entry. GCP will index these, so they can
        be used to improve the performance of log queries.
    """
    if isinstance(severity, str):
        # Validate severity.
        severity = Severity(severity)
    if len(fields) == 0:
        # No data was passed in to log, so its a noop.
        return False
    log_entry: t.Dict[str, t.Union[t.Dict[str, str], str]] = {"severity": severity.value}
    if trace_header and gcp_project_id:
        # Source: https://cloud.google.com/run/docs/logging#writing_structured_logs
        trace = trace_header.split("/")
        log_entry["logging.googleapis.com/trace"] = f"projects/{gcp_project_id}/traces/{trace[0]}"
    if labels:
        log_entry["logging.googleapis.com/labels"] = labels
    log_entry.update(fields)
    logger(json.dumps(log_entry))
    return True
