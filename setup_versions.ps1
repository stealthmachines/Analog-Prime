# Setup git branches and tags for historical versions
$WorkDir = "c:\Users\Owner\Documents\Conscious 2.0"
cd $WorkDir

$versions = @(
    @{num=1; branch="v1-initial"; msg="v1: Initial codebase"; tag="v1.0.0"},
    @{num=2; branch="v2-supplemental"; msg="v2: Codebase + supplemental"; tag="v2.0.0"},
    @{num=3; branch="v3-development"; msg="v3: Development iteration"; tag="v3.0.0"},
    @{num=4; branch="v4-features"; msg="v4: Feature additions"; tag="v4.0.0"},
    @{num=5; branch="v5-optimization"; msg="v5: Optimization pass"; tag="v5.0.0"},
    @{num=6; branch="v6-refinement"; msg="v6: Refinement and testing"; tag="v6.0.0"},
    @{num=7; branch="v7-stable"; msg="v7: Stable baseline"; tag="v7.0.0"}
)

foreach ($version in $versions) {
    Write-Host "`n=== Processing Version $($version.num) ===" -ForegroundColor Green
    
    $tempDir = "temp_v$($version.num)"
    $codebaseDir = "$tempDir\codebase"
    
    if (-not (Test-Path $codebaseDir)) {
        Write-Host "ERROR: $codebaseDir not found!" -ForegroundColor Red
        continue
    }
    
    # Create new branch
    git checkout -b $version.branch 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Created branch: $($version.branch)"
    } else {
        Write-Host "Branch already exists, checking out: $($version.branch)"
        git checkout $version.branch 2>&1 | Out-Null
    }
    
    # Copy codebase files
    Write-Host "Copying codebase files from $codebaseDir..."
    
    # Remove old codebase directory if it exists (but keep .git)
    if (Test-Path "$WorkDir\codebase") {
        Remove-Item "$WorkDir\codebase" -Recurse -Force -ErrorAction SilentlyContinue
    }
    
    # Copy new codebase
    Copy-Item "$codebaseDir\*" "$WorkDir\codebase" -Recurse -Force
    
    # Stage and commit
    git add -A
    $commitMsg = "$($version.msg)"
    git commit -m $commitMsg 2>&1 | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Committed: $commitMsg"
        
        # Create tag
        git tag -a $version.tag -m $version.msg 2>&1 | Out-Null
        Write-Host "Tagged: $($version.tag)"
    } else {
        Write-Host "No changes to commit for this version"
    }
    
    # Return to master for next iteration
    git checkout master 2>&1 | Out-Null
}

Write-Host "`n=== Version Setup Complete ===" -ForegroundColor Green
Write-Host "Listing all branches:" -ForegroundColor Yellow
git branch -a
Write-Host "`nListing all tags:" -ForegroundColor Yellow
git tag -l
