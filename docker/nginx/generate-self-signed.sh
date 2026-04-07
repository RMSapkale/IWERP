#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CERT_DIR="${SCRIPT_DIR}/certs"

mkdir -p "${CERT_DIR}"

openssl req -x509 -nodes -newkey rsa:2048 \
  -keyout "${CERT_DIR}/privkey.pem" \
  -out "${CERT_DIR}/fullchain.pem" \
  -days 365 \
  -subj "/CN=iwerp.com" \
  -addext "subjectAltName=DNS:iwerp.com,DNS:localhost,IP:127.0.0.1"

echo "Generated self-signed certs in ${CERT_DIR}"
