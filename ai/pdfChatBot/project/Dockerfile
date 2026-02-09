FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# COPY client.py .
COPY client-streamed.py .

# CMD ["python", "client.py"]
CMD ["python", "client-streamed.py"]
