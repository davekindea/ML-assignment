# Script to clean git history of large files
# This will remove large CSV and data files from all git history

Write-Host "Cleaning git history of large files..." -ForegroundColor Yellow
Write-Host "WARNING: This will rewrite git history!" -ForegroundColor Red
Write-Host ""

# Remove large files from all commits
$largeFiles = @(
    "regression/data/**/*.csv",
    "regression/results/**/*.png",
    "regression/results/**/*.json",
    "**/__pycache__/**"
)

# Use git filter-branch to remove files from history
Write-Host "Removing large files from git history..." -ForegroundColor Cyan

# Create a backup branch first
git branch backup-before-clean

# Remove files from all commits
git filter-branch --force --index-filter `
    "git rm --cached --ignore-unmatch -r regression/data/ regression/results/ **/__pycache__/" `
    --prune-empty --tag-name-filter cat -- --all

Write-Host ""
Write-Host "Cleaning up..." -ForegroundColor Cyan
git for-each-ref --format="delete %(refname)" refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

Write-Host ""
Write-Host "Done! Repository cleaned." -ForegroundColor Green
Write-Host "Check repository size with: git count-objects -vH" -ForegroundColor Yellow
Write-Host "If satisfied, push with: git push origin main --force" -ForegroundColor Yellow

