# Git Repository Cleanup Instructions

## Overview
This document provides step-by-step instructions for cleaning large files from git history using `git filter-repo`.

## Prerequisites
- ✅ Backup branch created: `backup-before-cleanup`
- ✅ `git-filter-repo` installed
- ✅ `.gitignore` updated to exclude large files

## Files to Remove from History

The following patterns will be removed:
- `venv/` - Python virtual environment (large libraries)
- `videos/*.mp4` - Video files
- `output/**/*.vtk` - VTK output files
- Any other large binary files

## Step-by-Step Process

### Step 1: Test on a Copy (SAFETY FIRST)

```bash
# Already done: Created test mirror
cd /workspace
git clone --mirror GPU-Accelerated-2D-Aerodynamics-via-Lattice-Boltzmann-using-CUDA/.git test-repo.git
cd test-repo.git

# Test the filter
git filter-repo --path venv/ --invert-paths --path videos/ --invert-paths --path output/ --invert-paths --force

# Check size reduction
du -sh .git
```

### Step 2: Apply to Main Repository

**ONLY proceed if test was successful!**

```bash
cd /workspace/GPU-Accelerated-2D-Aerodynamics-via-Lattice-Boltzmann-using-CUDA

# Remove large files from history
git filter-repo \
  --path venv/ --invert-paths \
  --path videos/ --invert-paths \
  --path output/ --invert-paths \
  --force

# Verify cleanup
git count-objects -vH
du -sh .git
```

### Step 3: Verify Results

```bash
# Check that large files are gone from history
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | awk '/^blob/ {if ($3 > 1048576) print $3/1048576 "MB", $4}' | sort -rn

# Should show minimal or no large files
```

### Step 4: Force Push to Remote (DESTRUCTIVE!)

**WARNING: This rewrites remote history. Only do this if:**
- ✅ Test was successful
- ✅ You're the only one working on this repo
- ✅ You have a backup

```bash
# Force push to remote
git push origin --force --all
git push origin --force --tags

# If you have a backup branch on remote, you may need to delete it first
# git push origin --delete backup-before-cleanup
```

### Step 5: Clean Up Local Repository

```bash
# Remove old remote tracking
git remote remove origin

# Re-add remote
git remote add origin https://github.com/shalofty/GPU-Accelerated-2D-Aerodynamics-via-Lattice-Boltzmann-using-CUDA.git

# Verify remote
git remote -v
```

## Rollback Instructions

If something goes wrong:

```bash
# Restore from backup branch
git checkout backup-before-cleanup
git branch -D main
git checkout -b main

# Or restore from remote (if you didn't force push yet)
git fetch origin
git reset --hard origin/main
```

## Expected Results

**Before:**
- Repository size: ~12GB (7.3GB .git)
- Pack files: ~7.2GB
- Large files in history: venv/, videos/, output/

**After (ACTUAL RESULTS):**
- ✅ Repository size: **187MB** (97% reduction!)
- ✅ Pack files: **185.73 MiB**
- ✅ Large files: **Completely removed from history**
- ✅ No files > 1MB remaining in history

## Notes

- This process rewrites ALL commit hashes
- Anyone with a local clone will need to re-clone
- The backup branch preserves the original state
- Test copy can be deleted after verification

