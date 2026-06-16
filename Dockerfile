FROM python:3.10-slim

WORKDIR /app

COPY flask_app/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader stopwords wordnet omw-1.4 punkt

COPY flask_app/ flask_app/
COPY models/ models/
COPY params.yaml params.yaml

ENV MODEL_SOURCE=local
ENV PYTHONPATH=/app

EXPOSE 5000

CMD ["python", "flask_app/app.py"]
