#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v python3 >/dev/null 2>&1; then
    printf '%s\n' 'Python 3.10+ is required to configure this lightweight package.' >&2
    exit 2
fi
exec python3 "$TASK_ROOT/install.py" "$@"
