FROM python:3.7.3-stretch

WORKDIR /app
ENV PYTHONPATH="/app/pythonpath"

COPY ringer /app/ringer
COPY setup.py README.md MANIFEST.in /app/

RUN pip install . \
    && mkdir -p $PYTHONPATH

COPY docker/ringer_config.py $PYTHONPATH/

CMD ["python", "-m", "ringer.main"]
