$OutputFile = "windows_output.log"

# Start the process
$Process = Start-Process -FilePath "pamalytics_windows_launcher.bat" -RedirectStandardOutput $OutputFile -RedirectStandardError "error.log" -PassThru

$Found = $false
$TimeoutSeconds = 60
$StartTime = Get-Date

try {
    while ((Get-Date) -lt $StartTime.AddSeconds($TimeoutSeconds)) {
        if (Test-Path $OutputFile) {
            $Content = Get-Content $OutputFile -Raw
            if ($Content -match "You can now view your Streamlit app in your browser.") {
                Write-Host "✅ Success message detected!"
                $Found = $true
                break
            }
        }
        Start-Sleep -Seconds 1
    }
}
finally {
    Write-Host "Manually stopping Streamlit process..."
    Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    # Also kill any orphaned python processes spawned by the batch file
    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

if (-not $Found) {
    Write-Host "❌ ERROR: Success message not found. Dumping logs..."
    Get-Content $OutputFile -ErrorAction SilentlyContinue
    Get-Content "error.log" -ErrorAction SilentlyContinue
    exit 1
}
