#!/usr/bin/env sh
set -e

ollama serve &
SERVER_PID=$!

# Ждём, пока поднимется API
echo "Waiting for ollama server..."
until curl -sS http://localhost:11434/api/tags >/dev/null 2>&1; do
  sleep 1
done

echo "Ollama is up, ensuring qwen3-vl:8b is available..."
ollama pull qwen3-vl:8b || true

echo "qwen3-vl:8b is ready. Keeping server running..."
wait "$SERVER_PID"