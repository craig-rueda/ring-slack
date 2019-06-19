ARG BASE_REPO=library
FROM ${BASE_REPO}/python:3.5-stretch

WORKDIR /app
ENV PYTHONPATH="/app/pythonpath"

COPY ringer /app/ringer
COPY setup.py README.md MANIFEST.in /app/
COPY docker/pip.conf /etc/

RUN pip install . \
    && mkdir -p $PYTHONPATH

COPY docker/ringer_config.py $PYTHONPATH/

CMD ["python", "-m", "ringer.main"]
