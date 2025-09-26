# auto-commit.ps1
cd "C:\Users\tyler\RedditVideoBot"

while ($true) {
    try {
        git add -A
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        git commit -m "Auto-commit $timestamp" | Out-Null
        git push origin main | Out-Null
        Write-Output "[$timestamp] Auto-commit + push done."
    }
    catch {
        Write-Output "[$(Get-Date -Format "HH:mm:ss")] Nothing new to commit."
    }
    Start-Sleep -Seconds 300  # 5 minutes
}
