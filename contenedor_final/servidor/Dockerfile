FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY servidor.py .
COPY model_unquant.tflite .
COPY labels.txt .

CMD ["python", "servidor.py"]
