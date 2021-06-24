FROM python:3.9.5-slim-buster

WORKDIR /app

COPY requirements-test.txt .
RUN pip install -r requirements-test.txt

COPY bavard_ml_common bavard_ml_common
COPY setup.py .
COPY README.md .
RUN pip install . && rm -r bavard_ml_common

COPY ./test ./test

ENTRYPOINT ["python", "-m", "unittest", "--verbose"]
