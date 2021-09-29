FROM python:3.9.5-slim-buster

WORKDIR /app

ENV POETRY_VERSION=1.1.10 \
    POETRY_NO_INTERACTION=1 \
    # Make poetry install to this location.
    POETRY_HOME="/opt/poetry" \
    # Make poetry create the virtual environment in the project's root. It gets named `.venv`.
    POETRY_VIRTUALENVS_IN_PROJECT=true

# Prepend poetry to $PATH.
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    # Needed for installing poetry.
    curl \
    # Needed for building python dependencies.
    build-essential

# Install poetry, respecting $POETRY_VERSION.
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

COPY poetry.lock pyproject.toml ./

# Install dependencies, including all extras.
RUN poetry install --extras "ml gcp"

COPY ./bavard_ml_common ./bavard_ml_common
COPY ./test ./test

CMD ["poetry", "run", "python", "-m", "unittest", "--verbose"]
