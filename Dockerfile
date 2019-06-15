FROM python:3.7.3-stretch

WORKDIR /app
ENV PYTHONPATH="/app"

COPY ringer /app/ringer
COPY setup.py README.md MANIFEST.in /app/

RUN pip install -e .

CMD ["python", "-m", "ringer.main"]
