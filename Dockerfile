FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application's source code and preserve folder structure
COPY ./src/codeinsightqa /app
COPY ./config /app/config
COPY ./data /app/data

EXPOSE 8000

CMD ["uvicorn", "qa_engine:app", "--host", "0.0.0.0", "--port", "8000"]