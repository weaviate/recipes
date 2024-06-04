#!/bin/bash

# Function to display usage instructions
usage() {
  echo "Usage: $0 --llm LLM_OPTION --key API_KEY"
  exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --llm) LLM="$2"; shift ;;
    --key) KEY="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; usage ;;
  esac
  shift
done

# Check if both arguments are provided
if [ -z "$LLM" ] || [ -z "$KEY" ]; then
  echo "Both --llm and --key arguments are required."
  usage
fi

# Set environment variables
export LLM="$LLM"
export KEY="$KEY"

# Print the set environment variables
echo "Set environment variable LLM to $LLM."
echo "Set environment variable KEY to $KEY."
