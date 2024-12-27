FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py .
COPY config/ config/
COPY models/ models/

CMD ["python", "main.py"]
