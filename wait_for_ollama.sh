#!/usr/bin/env sh
set -e

URL="$1"
shift

i=0
echo "Waiting for Ollama at $URL ..."
while ! curl -sS "$URL" >/dev/null 2>&1; do
  i=$((i+1))
  echo "  attempt $i..."
  if [ "$i" -ge 30 ]; then
    echo "Ollama did not become ready in time, continuing anyway..."
    break
  fi
  sleep 2
done

echo "Starting app: $@"
exec "$@"
