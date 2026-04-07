Place TLS certificates here for the edge proxy container.

Expected filenames:
- `fullchain.pem`
- `privkey.pem`

Options:
1. For local bring-up, generate self-signed files with:
   `./docker/nginx/generate-self-signed.sh`
2. For production, replace them with real certificates for `iwerp.com`.

This directory is mounted into the proxy container at:
- `/etc/nginx/certs/fullchain.pem`
- `/etc/nginx/certs/privkey.pem`
