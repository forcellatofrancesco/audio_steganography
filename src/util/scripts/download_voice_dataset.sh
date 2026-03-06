#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
INPUT_DIR="${REPO_ROOT}/input"
ARCHIVE_NAME="dev-clean.tar.gz"
ARCHIVE_PATH="${INPUT_DIR}/${ARCHIVE_NAME}"

CONFIG_PATH="${REPO_ROOT}/config.toml"
DATASET_URL="$(
python3 - "$CONFIG_PATH" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    cfg = tomllib.load(f)
print(cfg["dataset"]["voice_url"])
PY
)"
EXTRACTED_DIR_NAME="LibriSpeech"

mkdir -p "${INPUT_DIR}"

if command -v curl >/dev/null 2>&1; then
	echo "Downloading ${DATASET_URL} to ${ARCHIVE_PATH} ..."
	curl -fL "${DATASET_URL}" -o "${ARCHIVE_PATH}"
elif command -v wget >/dev/null 2>&1; then
	echo "Downloading ${DATASET_URL} to ${ARCHIVE_PATH} ..."
	wget -O "${ARCHIVE_PATH}" "${DATASET_URL}"
else
	echo "Error: neither 'curl' nor 'wget' is installed." >&2
	exit 1
fi

echo "Extracting ${ARCHIVE_PATH} into ${INPUT_DIR} ..."
tar -xzf "${ARCHIVE_PATH}" -C "${INPUT_DIR}"

if [[ -d "${INPUT_DIR}/${EXTRACTED_DIR_NAME}" ]]; then
	echo "Done. Extracted dataset is available at ${INPUT_DIR}/${EXTRACTED_DIR_NAME}"
else
	echo "Done extracting, but expected directory '${EXTRACTED_DIR_NAME}' was not found in ${INPUT_DIR}."
	echo "Please inspect extracted contents manually."
fi

rm $ARCHIVE_PATH