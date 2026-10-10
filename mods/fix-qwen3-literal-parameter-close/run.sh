#!/bin/bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$MOD_DIR/qwen3-literal-parameter-close.patch"
PREFIX="[fix-qwen3-literal-parameter-close]"

if ! command -v git >/dev/null 2>&1; then
  echo "$PREFIX git is required to apply this mod." >&2
  echo "$PREFIX Apply mods/use-official-vllm first if needed." >&2
  exit 1
fi

if [ ! -f "$PYTHON_ROOT/vllm/parser/qwen3.py" ]; then
  echo "$PREFIX $PYTHON_ROOT/vllm/parser/qwen3.py not found." >&2
  echo "$PREFIX This mod targets the engine-based Qwen3 parser (vLLM after #45413)." >&2
  exit 1
fi

cd "$PYTHON_ROOT"

if git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
  echo "$PREFIX Patch is already applied; skipping."
elif git apply --check "$PATCH_FILE"; then
  git apply "$PATCH_FILE"
  find vllm/parser -name "qwen3*.pyc" -delete 2>/dev/null || true
  echo "$PREFIX Applied: a literal </parameter> inside a tool-call value no longer truncates it."
else
  echo "$PREFIX Patch could not be applied to installed vLLM." >&2
  echo "$PREFIX Written against vLLM 6bbad6acdffd58b63037829f9fe27b92c6650071;" >&2
  echo "$PREFIX vllm/parser/qwen3.py has changed since, so the fix may be upstream now." >&2
  exit 1
fi
