FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ensure script is executable
RUN chmod +x /app/wait_for_ollama.sh

CMD ["./wait_for_ollama.sh", "http://ollama:11434/api/tags", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8081"]
