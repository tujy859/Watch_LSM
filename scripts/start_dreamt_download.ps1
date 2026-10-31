param(
    [string]$Username = 'tujy859',
    [string]$Version = '2.2.0',
    [string]$InterfaceAlias = 'WLAN'
)
$ErrorActionPreference = 'Stop'
$downloadRoot = Split-Path -Parent $PSScriptRoot
$downloadAdapter = Get-NetAdapter -Name $InterfaceAlias
if ($downloadAdapter.Status -ne 'Up' -or -not $downloadAdapter.HardwareInterface) {
    throw 'Select an active physical network adapter; a VPN/TUN adapter is not allowed.'
}
$downloadIp = (Get-NetIPAddress -InterfaceIndex $downloadAdapter.ifIndex -AddressFamily IPv4 | Select-Object -First 1).IPAddress
if (-not $downloadIp) { throw 'The physical adapter has no IPv4 address.' }
$host.UI.RawUI.WindowTitle = "DREAMT $Version - Direct physical network"
Write-Host 'The password is entered only in this window and is not stored in chat or files.'
Write-Host 'Keep this window open until the download completes. Verified files are reused on restart.'
& (Join-Path $downloadRoot '.local/physionet-venv/Scripts/python.exe') -u (Join-Path $downloadRoot 'scripts/download_dreamt_direct.py') --username $Username --version $Version --source-ip $downloadIp --interface-index $downloadAdapter.ifIndex
