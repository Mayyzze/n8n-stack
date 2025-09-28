param(
    [string]$Context = "mypi"
)

$compose = "docker --context $Context compose -f docker-compose.yml"

Write-Output "[+] Building local images (custom n8n)..."
Invoke-Expression "$compose build n8n"

Write-Output "[+] Pulling remote images (traefik, etc.)..."
Invoke-Expression "$compose pull traefik"

Write-Output "[+] Starting stack..."
Invoke-Expression "$compose up -d"

Write-Output "[+] Showing status..."
Invoke-Expression "$compose ps"