#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "Missing .venv. Create it with: python -m venv .venv"
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/chatbot/train.py" ]]; then
  echo "chatbot/train.py not found"
  exit 1
fi

echo "Ensuring NLTK resources (punkt, stopwords)..."
"${VENV_PY}" -m nltk.downloader punkt stopwords -d "${ROOT_DIR}/chatbot/nltk_data"

echo "Starting model training..."
exec "${VENV_PY}" "${ROOT_DIR}/chatbot/train.py"
