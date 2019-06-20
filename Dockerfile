ARG BASE_REPO=library
FROM ${BASE_REPO}/python:3.5-jessie

WORKDIR /app
ENV PYTHONPATH="/app/pythonpath"

COPY ringer /app/ringer
COPY setup.py README.md MANIFEST.in /app/
COPY docker/pip.conf /etc/

RUN apt-get update && apt-get install -y python-opencv libgtk-3-dev \
    && pip install . \
    && mkdir -p $PYTHONPATH \
    && rm -rf /var/lib/apt/lists/*

COPY docker/ringer_config.py $PYTHONPATH/

CMD ["python", "-m", "ringer.main"]
