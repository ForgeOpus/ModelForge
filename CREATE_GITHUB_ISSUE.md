# How to Create the GitHub Issue for This Bug Fix

This document provides instructions for creating a GitHub issue to track this bug fix.

## Option 1: Using GitHub Web Interface

1. Navigate to: https://github.com/RETR0-OS/ModelForge/issues/new
2. Click "Bug report" template
3. Use the following content:

### Title
```
🐛 Critical Bug: Database Schema Mismatch Causes Fine-Tuning to Fail [FIXED]
```

### Body
Copy the content from `ISSUE_DATABASE_SCHEMA_MISMATCH.md` or use this summary:

```markdown
**Status:** ✅ FIXED in PR #[PR_NUMBER]

## Problem
The fine-tuning router attempted to save an `is_custom_base_model` field to the database, but this field was missing from the database schema. This caused:
- Database insertion failures
- Fine-tuned models not appearing in the UI
- Custom models being created on disk but not tracked in the database

## Solution
1. ✅ Added `is_custom_base_model` BOOLEAN column to database schema
2. ✅ Implemented automatic migration for existing databases
3. ✅ Updated INSERT statements to include new field
4. ✅ Ensured backward compatibility with zero data loss

## Testing
- ✅ Tested new database creation with updated schema
- ✅ Tested automatic migration from old schema to new schema
- ✅ Verified custom and recommended models save correctly
- ✅ Confirmed backward compatibility maintained
- ✅ No data loss during migration

## Files Changed
- `ModelForge/utilities/settings_managers/DBManager.py`

See `ISSUE_DATABASE_SCHEMA_MISMATCH.md` for complete technical documentation.
```

### Labels
- `bug`
- `database`
- `fixed`
- `high-priority`

## Option 2: Using GitHub CLI (gh)

If you have the `gh` CLI tool installed and configured:

```bash
cd /home/runner/work/ModelForge/ModelForge
gh issue create \
  --title "🐛 Critical Bug: Database Schema Mismatch Causes Fine-Tuning to Fail [FIXED]" \
  --body-file ISSUE_DATABASE_SCHEMA_MISMATCH.md \
  --label "bug,database,fixed,high-priority"
```

## Option 3: Using GitHub API

If you have a GitHub personal access token:

```bash
export GITHUB_TOKEN="your_token_here"

curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/RETR0-OS/ModelForge/issues \
  -d @- << 'EOF'
{
  "title": "🐛 Critical Bug: Database Schema Mismatch Causes Fine-Tuning to Fail [FIXED]",
  "body": "See ISSUE_DATABASE_SCHEMA_MISMATCH.md in the repository for complete details.",
  "labels": ["bug", "database", "fixed", "high-priority"]
}
EOF
```

## Why This Issue Matters

This bug fix addresses a **critical** issue that:
- Broke the custom model fine-tuning feature
- Prevented users from tracking their fine-tuned models
- Created silent failures that confused users
- Required immediate attention

The fix includes:
- Database schema update
- Automatic migration for existing databases
- Comprehensive testing
- Full backward compatibility

## Related PR

This issue is resolved in the PR that includes these commits:
- `Fix critical database schema mismatch for custom models`
- `Add comprehensive documentation for database schema fix`
- `Fix documentation reference in issue document`

## Next Steps

After creating the issue:
1. Link it to the PR
2. Close it with a comment referencing the fix commit
3. Consider adding it to a milestone if you track releases
