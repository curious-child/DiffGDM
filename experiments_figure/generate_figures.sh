#!/usr/bin/env bash
set -euo pipefail

# This wrapper only dispatches the original, result-reading plotting functions.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${DIFFGDM_FIGURE_PYTHON:-python}"
ARGS=()
while (( $# )); do
    case "$1" in
        --python)
            if (( $# < 2 )) || [[ "$2" == --* ]]; then
                printf '%s\n' 'error: --python requires an executable path' >&2
                exit 2
            fi
            PYTHON_BIN="$2"
            shift 2
            ;;
        --python=*) PYTHON_BIN="${1#--python=}"; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
if [[ -z "$PYTHON_BIN" ]] || ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf '%s\n' 'error: Python not found; use --python /path/to/python (no environment is installed automatically)' >&2
    exit 1
fi
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
exec "$PYTHON_BIN" -B "$SCRIPT_DIR/generate_figures.py" "${ARGS[@]}"
