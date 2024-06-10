#!/bin/bash

usage() {
  echo "Usage: $0 --llm LLM_OPTION --key API_KEY"
  exit 1
}

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --llm) LLM="$2"; shift ;;
    --key) KEY="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; usage ;;
  esac
  shift
done

if [ -z "$LLM" ] || [ -z "$KEY" ]; then
  echo "Both --llm and --key arguments are required."
  usage
fi

export LLM="$LLM"
export KEY="$KEY"

echo "Set environment variable LLM to $LLM."
echo "Set environment variable KEY to $KEY."

docker compose up -d

echo "Waiting for 5 seconds to allow the Docker container to start..."
sleep 5

echo "Restoring backup..."
python3 backup_manager.py --restore
echo "Backup restored, waiting 5 seconds to start backend..."
sleep 5

uvicorn main:app --reload