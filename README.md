# n8n-stack (Raspberry Pi)

## One-off
1. docker context create mypi --docker "host=ssh://mayyzze@<PI_IP>"
2. cp .env.sample .env  # ajuster TZ si besoin

## Deploy
- make deploy
- ouvrir http://<PI_IP>/

## Logs
- make logs

## Volumes — TL;DR
- `n8n_data` est un volume nommé → persistance sur la Pi, indépendant des chemins locaux.
- Backup rapide :
  docker run --rm -v n8n_data:/data alpine tar czf - /data > n8n_data_backup.tgz
- Restore :
  cat n8n_data_backup.tgz | docker run --rm -i -v n8n_data:/data alpine tar xzf - -C /data