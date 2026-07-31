#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 https://github.com/USERNAME/REPOSITORY.git" >&2
  exit 2
fi

git remote remove origin 2>/dev/null || true
git remote add origin "$1"
git push -u origin main
