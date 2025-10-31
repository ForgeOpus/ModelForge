#!/bin/bash
# Script to create the Bug Fix and Implementation Improvements GitHub Issue
# Usage: ./create_bug_fix_issue.sh

set -e

REPO_OWNER="RETR0-OS"
REPO_NAME="ModelForge"
ISSUE_TITLE="🐛 Bug Fixes and Implementation Improvements Plan"
ISSUE_FILE="BUG_FIX_PLAN.md"

echo "Creating GitHub issue for Bug Fixes and Implementation Improvements..."
echo "Repository: $REPO_OWNER/$REPO_NAME"
echo "Title: $ISSUE_TITLE"
echo ""

# Check if gh CLI is available and authenticated
if command -v gh &> /dev/null; then
    if gh auth status &> /dev/null; then
        echo "Using GitHub CLI to create issue..."
        gh issue create \
          --repo "$REPO_OWNER/$REPO_NAME" \
          --title "$ISSUE_TITLE" \
          --body-file "$ISSUE_FILE" \
          --label "bug,enhancement,priority:high"
        
        echo "✅ Issue created successfully!"
        exit 0
    else
        echo "⚠️  GitHub CLI is installed but not authenticated."
        echo "Please run: gh auth login"
    fi
else
    echo "⚠️  GitHub CLI (gh) is not installed."
    echo "Please install it from: https://cli.github.com/"
fi

echo ""
echo "Alternative: Create the issue manually"
echo "1. Go to: https://github.com/$REPO_OWNER/$REPO_NAME/issues/new"
echo "2. Title: $ISSUE_TITLE"
echo "3. Copy content from: $ISSUE_FILE"
echo "4. Add labels: bug, enhancement, priority:high"
echo ""
echo "For more details, see: CREATE_ISSUE_INSTRUCTIONS.md"

