param(
    [string]$Context = "mypi"
)

$compose = "docker --context $Context compose -f docker-compose.yml"

Write-Output "[+] Pulling images..."
Invoke-Expression "$compose pull"

Write-Output "[+] Starting stack..."
Invoke-Expression "$compose up -d"

Write-Output "[+] Showing status..."
Invoke-Expression "$compose ps"