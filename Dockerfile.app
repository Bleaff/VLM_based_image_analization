FROM python:3.11-slim

WORKDIR /app

# Установим системные зависимости (curl для health check)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY wait_for_ollama.sh /usr/local/bin/wait_for_ollama.sh
RUN chmod +x /usr/local/bin/wait_for_ollama.sh

CMD ["./wait_for_ollama.sh", "http://ollama:11434/api/generate", "uvicorn pipeline_app:app --host 0.0.0.0 --port 8081 --reload"]
