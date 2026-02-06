#!/bin/bash

# 1. Initialize Git
echo "Initializing new Git repository..."
git init

# 2. Add Files
echo "Adding files (Code and Documentation)..."
git add assignment_3.py README.md .gitignore

# 3. Commit
echo "Committing files..."
git commit -m "Completed Lab 03: Custom k-NN and Matrix Inversion Implementation"

# 4. Rename branch to main
git branch -M main

# 5. Prompt for Remote URL
echo ""
echo "--------------------------------------------------------"
echo "STEP REQUIRED: Create a new repository on GitHub now."
echo "Link: https://github.com/new"
echo "IMPORTANT: Do NOT check 'Initialize with README' or .gitignore."
echo "--------------------------------------------------------"
echo ""
read -p "Paste your new repository URL here (e.g., https://github.com/user/lab03.git): " repo_url

if [ -z "$repo_url" ]; then
    echo "Error: URL cannot be empty. Please run the script again."
    exit 1
fi

# 6. Add Remote and Push
echo "Adding remote origin..."
# Remove origin if it exists to avoid errors on re-runs
git remote remove origin 2>/dev/null
git remote add origin "$repo_url"

echo "Pushing code to GitHub..."
git push -u origin main

echo ""
echo "Done! Your code is now live at: $repo_url"

