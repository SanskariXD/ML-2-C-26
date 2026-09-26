#!/usr/bin/env bash
# Run only after reviewing the files and authenticating GitHub CLI locally.
# Private create-only. Never targets or modifies the Plan 1 repository.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v gh >/dev/null || { echo "Install GitHub CLI and run gh auth login first." >&2; exit 1; }
gh auth status >/dev/null
OWNER=$(gh api user --jq .login)
NAME="ML-2-C-26"
if [[ "$OWNER" != "SanskariXD" ]]; then
  echo "Authenticated as $OWNER; expected SanskariXD. No repository was created." >&2
  exit 1
fi
if gh repo view "$OWNER/$NAME" >/dev/null 2>&1; then
  echo "$OWNER/$NAME already exists. Refusing to modify it automatically." >&2
  exit 1
fi
TOP=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [[ -n "$TOP" && "$TOP" != "$(pwd -P)" ]]; then
  echo "This folder is inside another Git checkout. Move it out before publishing." >&2
  exit 1
fi
if [[ ! -d .git ]]; then
  git init -b main
fi
if git remote get-url origin >/dev/null 2>&1; then
  echo "Existing origin detected; refusing to replace it." >&2
  exit 1
fi
git var GIT_AUTHOR_IDENT >/dev/null || { echo "Configure your git user.name and user.email first." >&2; exit 1; }
git add .gitignore .gitattributes .github README.md WORKFLOW.md docs ml2 tests scripts notebooks evidence reference requirements.txt pyproject.toml
if ! git diff --cached --quiet; then
  git commit -m "Start Plan 2: audited baseline, CARE prototype and correctness tests"
fi
# GitHub refuses creation if the repository exists or if scope/network checks fail.
gh repo create "$OWNER/$NAME" --private --source=. --remote=origin --push \
  --description "ML 2 C 26 — independent, auditable business entity resolution track"
gh repo view "$OWNER/$NAME" --json nameWithOwner,isPrivate,url
