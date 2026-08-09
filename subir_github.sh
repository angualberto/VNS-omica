#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/andre/423ff97c-bf80-4bc9-9621-cf3ec081fd25/iaimgem"
REPO="https://github.com/angualberto/VNS-omica.git"

if ! command -v git >/dev/null 2>&1; then
    echo "Instale o git: sudo apt-get install -y git" >&2
    exit 1
fi

cd "$ROOT"

if [ ! -d .git ]; then
    git init -b main
fi

git add .
git status --short | head -40
echo
echo "Total de arquivos a subir: $(git status --short | wc -l)"

if git remote | grep -q origin; then
    git remote set-url origin "$REPO"
else
    git remote add origin "$REPO"
fi

read -rp "Confirmar push? (sim/nao): " ok
if [ "$ok" = "sim" ]; then
    git commit -m "primeiro commit"
    git push -u origin main
else
    echo "Abortado. Confira o git status acima."
fi