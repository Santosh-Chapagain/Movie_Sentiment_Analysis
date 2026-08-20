FROM python:3.10-slim

WORKDIR /app

COPY app/ /app/app/
COPY src/ /app/src/
COPY models/ /app/models/
COPY app/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet punkt punkt_tab

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "-k", "uvicorn.workers.UvicornWorker", "app.app:app"]