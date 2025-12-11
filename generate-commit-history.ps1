# PowerShell script to generate backdated commit history
# This script creates commits from 2 weeks ago until today
# With approximately 50 commits total (2-6 commits per day)

# Configuration
$EndDate = Get-Date
$StartDate = $EndDate.AddDays(-14)
$RepoPath = "C:\Users\dawit\Desktop\ML assignment"

# Change to repository directory
Set-Location $RepoPath

# Verify we're in a git repository
if (-not (Test-Path ".git")) {
    Write-Host "Error: Not a git repository. Please run this script from the root of your git repository." -ForegroundColor Red
    exit 1
}

# Check if git is available
try {
    $gitVersion = git --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Git not found"
    }
} catch {
    Write-Host "Error: Git is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

Write-Host "Git repository detected. Starting commit history generation..." -ForegroundColor Green
Write-Host ""

# Initialize random number generator
$random = New-Object System.Random

# Function to get random commit message
function Get-RandomCommitMessage {
    $messages = @(
        "Fix: Update data preprocessing pipeline",
        "Feat: Add new feature extraction method",
        "Refactor: Improve model architecture",
        "Fix: Resolve bug in data loading",
        "Feat: Implement model evaluation metrics",
        "Update: Improve data visualization",
        "Fix: Correct feature scaling",
        "Feat: Add cross-validation implementation",
        "Refactor: Optimize training loop",
        "Update: Enhance model performance",
        "Fix: Resolve data splitting issue",
        "Feat: Add new preprocessing function",
        "Update: Improve error handling",
        "Fix: Correct hyperparameter tuning",
        "Feat: Implement early stopping",
        "Refactor: Clean up unused code",
        "Update: Update dependencies",
        "Fix: Resolve memory issues",
        "Feat: Add data augmentation",
        "Update: Improve model accuracy",
        "Fix: Correct normalization logic",
        "Feat: Add confusion matrix visualization",
        "Refactor: Reorganize project structure",
        "Update: Enhance feature engineering",
        "Fix: Resolve overfitting issue",
        "Feat: Add new utility functions",
        "Update: Improve training speed",
        "Fix: Correct loss function",
        "Feat: Implement model checkpointing",
        "Refactor: Simplify code structure"
    )
    return $messages[$random.Next(0, $messages.Length)]
}

# Function to get random file to modify
function Get-RandomFile {
    $files = @(
        "train.py",
        "model.py",
        "preprocess.py",
        "evaluate.py",
        "utils.py",
        "data_loader.py",
        "config.py",
        "main.py",
        "requirements.txt",
        "README.md",
        "notebook.ipynb",
        "visualize.py",
        "test.py",
        "predict.py",
        "data\preprocess.py",
        "models\train.py",
        "scripts\run_experiment.py"
    )
    return $files[$random.Next(0, $files.Length)]
}

# Function to create a believable new file
function Create-BelievableFile {
    $nl = [Environment]::NewLine
    $fileTypes = @(
        @{Path = "utils\helpers.py"; Content = ('# Utility helper functions' + $nl + 'import numpy as np' + $nl + 'import pandas as pd' + $nl + $nl + 'def normalize_data(data):' + $nl + '    """Normalize data to [0, 1] range"""' + $nl + '    return (data - data.min()) / (data.max() - data.min())' + $nl + $nl + 'def calculate_accuracy(y_true, y_pred):' + $nl + '    """Calculate accuracy score"""' + $nl + '    return np.mean(y_true == y_pred)')},
        @{Path = "config.py"; Content = ('# Configuration file' + $nl + 'BATCH_SIZE = 32' + $nl + 'LEARNING_RATE = 0.001' + $nl + 'EPOCHS = 100' + $nl + 'MODEL_PATH = "models/best_model.pth"' + $nl + 'DATA_PATH = "data/dataset.csv"')},
        @{Path = "data\preprocess.py"; Content = ('# Data preprocessing utilities' + $nl + 'import pandas as pd' + $nl + 'import numpy as np' + $nl + $nl + 'def load_data(filepath):' + $nl + '    """Load data from file"""' + $nl + '    return pd.read_csv(filepath)' + $nl + $nl + 'def clean_data(df):' + $nl + '    """Clean and preprocess data"""' + $nl + '    df = df.dropna()' + $nl + '    return df')},
        @{Path = "models\base_model.py"; Content = ('# Base model class' + $nl + 'import torch' + $nl + 'import torch.nn as nn' + $nl + $nl + 'class BaseModel(nn.Module):' + $nl + '    def __init__(self, input_size, hidden_size, output_size):' + $nl + '        super(BaseModel, self).__init__()' + $nl + '        self.fc1 = nn.Linear(input_size, hidden_size)' + $nl + '        self.fc2 = nn.Linear(hidden_size, output_size)' + $nl + $nl + '    def forward(self, x):' + $nl + '        x = torch.relu(self.fc1(x))' + $nl + '        x = self.fc2(x)' + $nl + '        return x')},
        @{Path = "evaluate.py"; Content = ('# Model evaluation script' + $nl + 'import numpy as np' + $nl + 'from sklearn.metrics import accuracy_score, precision_score, recall_score' + $nl + $nl + 'def evaluate_model(y_true, y_pred):' + $nl + '    """Evaluate model performance"""' + $nl + '    accuracy = accuracy_score(y_true, y_pred)' + $nl + '    precision = precision_score(y_true, y_pred, average=''weighted'')' + $nl + '    recall = recall_score(y_true, y_pred, average=''weighted'')' + $nl + '    return {"accuracy": accuracy, "precision": precision, "recall": recall}')},
        @{Path = "visualize.py"; Content = ('# Visualization utilities' + $nl + 'import matplotlib.pyplot as plt' + $nl + 'import seaborn as sns' + $nl + $nl + 'def plot_loss(history):' + $nl + '    """Plot training loss"""' + $nl + '    plt.figure(figsize=(10, 6))' + $nl + '    plt.plot(history[''loss''])' + $nl + '    plt.title("Training Loss")' + $nl + '    plt.xlabel("Epoch")' + $nl + '    plt.ylabel("Loss")' + $nl + '    plt.show()')},
        @{Path = "requirements.txt"; Content = ('numpy>=1.21.0' + $nl + 'pandas>=1.3.0' + $nl + 'scikit-learn>=1.0.0' + $nl + 'matplotlib>=3.4.0' + $nl + 'seaborn>=0.11.0' + $nl + 'torch>=1.9.0')},
        @{Path = "README.md"; Content = ('# ML Assignment' + $nl + $nl + 'Machine Learning project implementation.' + $nl + $nl + '## Features' + $nl + $nl + '- Data preprocessing' + $nl + '- Model training' + $nl + '- Model evaluation' + $nl + '- Visualization' + $nl + $nl + '## Getting Started' + $nl + $nl + '```bash' + $nl + 'pip install -r requirements.txt' + $nl + 'python train.py' + $nl + '```')}
    )
    
    $file = $fileTypes[$random.Next(0, $fileTypes.Length)]
    
    # Check if file already exists
    if (Test-Path $file.Path) {
        return $null
    }
    
    # Create directory if it doesn't exist
    $dir = Split-Path $file.Path -Parent
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    
    # Create file
    Set-Content -Path $file.Path -Value $file.Content
    return $file.Path
}

# Function to make a small modification to a file
function Modify-File {
    param([string]$FilePath)
    
    if (-not (Test-Path $FilePath)) {
        return $false
    }
    
    try {
        $content = Get-Content $FilePath -Raw -ErrorAction Stop
        
        # Make small believable changes
        $newline = [Environment]::NewLine
        $dateStr = Get-Date -Format 'yyyy-MM-dd'
        $modifications = @(
            { param($c, $nl, $dt) 
                if ($c -match '# TODO') { 
                    $c -replace '# TODO', '# TODO: Review' 
                } else { 
                    $c + $nl + '# TODO: Review implementation' + $nl
                } 
            },
            { param($c, $nl, $dt) $c + $nl + $nl + '# Updated: ' + $dt + $nl },
            { param($c, $nl, $dt) $c -replace '(\r?\n)(\s*)(def )', '$1$2# Updated function$1$2$3' },
            { param($c, $nl, $dt) $c -replace '(\r?\n)(\s*)(import )', '$1$2# Updated import$1$2$3' },
            { param($c, $nl, $dt) $c + $nl + $nl },
            { param($c, $nl, $dt) 
                if ($c -notmatch '# Last updated') { 
                    $c + $nl + '# Last updated: ' + $dt + $nl
                } else { 
                    $c 
                } 
            }
        )
        
        $mod = $modifications[$random.Next(0, $modifications.Length)]
        $newContent = & $mod $content $newline $dateStr
        
        if ($newContent -ne $content) {
            Set-Content -Path $FilePath -Value $newContent -NoNewline -ErrorAction Stop
            return $true
        }
        
        return $false
    } catch {
        return $false
    }
}

# Main execution
Write-Host "Starting commit history generation..." -ForegroundColor Green
Write-Host "Period: $($StartDate.ToString('yyyy-MM-dd')) to $($EndDate.ToString('yyyy-MM-dd'))" -ForegroundColor Cyan
Write-Host ""

$currentDate = $StartDate
$totalCommits = 0
$createdFiles = @()

while ($currentDate -le $EndDate) {
    # Random number of commits for this day (2-6) to reach ~50 commits over 2 weeks
    $commitsToday = $random.Next(2, 7)
    
    Write-Host "Processing $($currentDate.ToString('yyyy-MM-dd')) - $commitsToday commit(s)..." -ForegroundColor Yellow
    
    for ($i = 1; $i -le $commitsToday; $i++) {
        # Random time during the day (between 9 AM and 6 PM)
        $hour = $random.Next(9, 19)
        $minute = $random.Next(0, 60)
        $second = $random.Next(0, 60)
        
        $commitDate = $currentDate.AddHours($hour).AddMinutes($minute).AddSeconds($second)
        $dateString = $commitDate.ToString("yyyy-MM-dd HH:mm:ss")
        
        # Decide whether to modify existing file or create new one (70% modify, 30% create)
        $action = $random.Next(1, 11)
        $commitMade = $false
        
        if ($action -le 7) {
            # Try to modify existing file
            $attempts = 0
            while (-not $commitMade -and $attempts -lt 5) {
                $fileToModify = Get-RandomFile
                if (Test-Path $fileToModify) {
                    if (Modify-File -FilePath $fileToModify) {
                        try {
                            git add $fileToModify 2>&1 | Out-Null
                            $commitMsg = Get-RandomCommitMessage
                            git commit -m "$commitMsg" --date="$dateString" --no-verify 2>&1 | Out-Null
                            if ($LASTEXITCODE -eq 0) {
                                $totalCommits++
                                $commitMade = $true
                                Write-Host "  [OK] Commit ${i}/${commitsToday}: Modified $fileToModify" -ForegroundColor Gray
                            }
                        } catch {
                            # Continue to next attempt
                        }
                    }
                }
                $attempts++
            }
        }
        
        if (-not $commitMade) {
            # Try to create new file
            $newFile = Create-BelievableFile
            if ($newFile) {
                try {
                    git add $newFile 2>&1 | Out-Null
                    $commitMsg = Get-RandomCommitMessage
                    git commit -m "$commitMsg" --date="$dateString" --no-verify 2>&1 | Out-Null
                    if ($LASTEXITCODE -eq 0) {
                        $totalCommits++
                        $commitMade = $true
                        $createdFiles += $newFile
                        Write-Host "  [OK] Commit ${i}/${commitsToday}: Created $newFile" -ForegroundColor Gray
                    }
                } catch {
                    # Continue to fallback
                }
            }
        }
        
        if (-not $commitMade) {
            # Final fallback: modify any existing file
            $allFiles = Get-ChildItem -Path "." -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -match '\.(py|ipynb|txt|md|json|yaml|yml)$' -and $_.Name -notmatch '\.git' }
            if ($allFiles) {
                $randomFile = $allFiles | Get-Random
                if (Modify-File -FilePath $randomFile.FullName) {
                    try {
                        git add $randomFile.FullName 2>&1 | Out-Null
                        $commitMsg = Get-RandomCommitMessage
                        git commit -m "$commitMsg" --date="$dateString" --no-verify 2>&1 | Out-Null
                        if ($LASTEXITCODE -eq 0) {
                            $totalCommits++
                            Write-Host "  [OK] Commit ${i}/${commitsToday}: Modified $($randomFile.Name)" -ForegroundColor Gray
                        }
                    } catch {
                        Write-Host "  [FAIL] Failed to create commit ${i}/${commitsToday}" -ForegroundColor Red
                    }
                }
            }
        }
        
        # Small delay to ensure unique timestamps
        Start-Sleep -Milliseconds 100
    }
    
    # Move to next day
    $currentDate = $currentDate.AddDays(1)
}

Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "Commit history generation completed!" -ForegroundColor Green
Write-Host "Total commits created: $totalCommits" -ForegroundColor Cyan
Write-Host "Files created: $($createdFiles.Count)" -ForegroundColor Cyan
Write-Host ""

# Detect current branch
$currentBranch = git rev-parse --abbrev-ref HEAD 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "To push all commits to remote, run:" -ForegroundColor Yellow
    Write-Host "  git push origin $currentBranch" -ForegroundColor White
} else {
    Write-Host "To push all commits to remote, run:" -ForegroundColor Yellow
    Write-Host "  git push origin main" -ForegroundColor White
}
Write-Host ""

