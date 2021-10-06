FROM python:3.9.5-slim-buster as base
WORKDIR /app
ENV POETRY_VERSION=1.1.10 \
    POETRY_NO_INTERACTION=1 \
    POETRY_HOME="/opt/poetry" \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv" \
    POETRY_VIRTUALENVS_IN_PROJECT=true
# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

FROM base as builder-base
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl build-essential
# Install poetry, respecting $POETRY_VERSION.
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
WORKDIR $PYSETUP_PATH
COPY poetry.lock pyproject.toml ./
# Install dependencies, including dev dependencies and all extras.
RUN poetry install --extras "ml gcp"

FROM base as test
# Bring in the dependencies.
COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH
WORKDIR /app
COPY ./bavard_ml_utils ./bavard_ml_utils
COPY ./test ./test
CMD ["python", "-m", "unittest", "--verbose"]

FROM base as docs
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl build-essential \
    && curl -sL https://deb.nodesource.com/setup_16.x | bash - \
    && apt-get install --no-install-recommends -y nodejs \
    && npm install -g npm \
    && npm install -g firebase-tools
# Bring in the dependencies.
COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH
WORKDIR /app
COPY ./bavard_ml_utils ./bavard_ml_utils
COPY ./docs ./docs
COPY ./pyproject.toml ./
COPY ./scripts/build-docs.sh ./scripts/build-docs.sh
RUN ./scripts/build-docs.sh
WORKDIR /app/docs
CMD firebase deploy --token "$FIREBASE_TOKEN" --project bavard-dev --only hosting:docs-bavard-ml-utils
