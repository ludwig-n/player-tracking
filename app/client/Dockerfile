FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV SERVER_URL="http://server:8500"
CMD ["python", "-m", "streamlit", "run", "main.py"]
