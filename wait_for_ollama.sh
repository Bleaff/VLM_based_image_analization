#!/usr/bin/env bash

URL="$1"
shift

i=0
until curl -sS -X POST -H "Content-Type: application/json" -d '{"model":"__healthcheck","prompt":"ping","max_tokens":1}' "$URL" >/dev/null 2>&1; do
  i=$((i+1))
  echo "Waiting for Ollama at $URL ($i/30)..."
  if [ "$i" -ge 30 ]; then
    echo "Ollama did not become ready in time, continuing anyway..."
    break
  fi
  sleep 2
done
# exec the rest (start uvicorn)
exec "$@"
