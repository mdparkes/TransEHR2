#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# update_manifest.sh
#
# Updates an existing entry in manifest.csv. Recomputes the sha256 checksum
# and overwrites the corresponding row.
#
# This script will NOT add new entries. New entries must be added to
# manifest.csv by hand. This prevents data files that cannot be legally
# shared from being inadvertently registered for distribution.
#
# Usage:
#   ./update_manifest.sh <file_path> <source> <source_type> [DATA_ROOT]
#
# Arguments:
#   file_path     Path to the file. Can be:
#                   - absolute
#                   - relative to DATA_ROOT
#                   - relative to the current working directory
#   source        Where to fetch the file from. Either a filesystem path
#                 (for local_copy / local_symlink) or a URL (for download).
#   source_type   One of: local_copy, local_symlink, download
#   DATA_ROOT     Optional. Overrides SHARED_DATA_ROOT and the default.
#
# Environment variables:
#   SHARED_DATA_ROOT   Path to the data root directory.
#
# Examples:
#   # Update checksum after a file changes
#   ./update_manifest.sh ibd/RMT23345/encounters.parquet \
#       /home/michael/TransEHR2-IBDreadmission/data/ibd/RMT23345/encounters.parquet \
#       local_copy
#
#   # Update source URL for a downloadable resource
#   ./update_manifest.sh resources/atc_codes.csv \
#       https://example.org/atc_codes.csv \
#       download
# -----------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
DEFAULT_DATA_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)/data"

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <file_path> <source> <source_type> [DATA_ROOT]" >&2
    exit 1
fi

RAW_PATH="$1"
SOURCE="$2"
SOURCE_TYPE="$3"

if [[ $# -ge 4 ]]; then
    DATA_ROOT="$(realpath -m "$4")"
elif [[ -n "${SHARED_DATA_ROOT:-}" ]]; then
    DATA_ROOT="$(realpath -m "$SHARED_DATA_ROOT")"
else
    DATA_ROOT="$DEFAULT_DATA_ROOT"
fi

MANIFEST="$PROJECT_ROOT/manifest.csv"

# ---------------------------------------------------------------------------
# Validate source_type
# ---------------------------------------------------------------------------

case "$SOURCE_TYPE" in
    local_copy|local_symlink|download) ;;
    *)
        echo "Error: source_type must be one of: local_copy, local_symlink, download" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Resolve the file to an absolute path, then derive rel_path from DATA_ROOT
# ---------------------------------------------------------------------------

# Try interpreting RAW_PATH as: absolute → relative to DATA_ROOT → relative to cwd
if [[ -e "$RAW_PATH" ]]; then
    ABS_PATH="$(realpath "$RAW_PATH")"
elif [[ -e "$DATA_ROOT/$RAW_PATH" ]]; then
    ABS_PATH="$(realpath "$DATA_ROOT/$RAW_PATH")"
else
    echo "Error: File not found: $RAW_PATH" >&2
    echo "       Looked at: $RAW_PATH and $DATA_ROOT/$RAW_PATH" >&2
    exit 1
fi

# Derive the manifest path (relative to DATA_ROOT)
if [[ "$ABS_PATH" == "$DATA_ROOT"/* ]]; then
    REL_PATH="${ABS_PATH#"$DATA_ROOT/"}"
else
    echo "Error: File is not under DATA_ROOT ($DATA_ROOT)." >&2
    echo "       Resolved path: $ABS_PATH" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

sha256_of() {
    if command -v sha256sum &>/dev/null; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

# ---------------------------------------------------------------------------
# Compute checksum (skip for symlinks)
# ---------------------------------------------------------------------------

if [[ "$SOURCE_TYPE" == "local_symlink" ]]; then
    CHECKSUM="n/a"
else
    echo "Computing checksum for: $ABS_PATH"
    CHECKSUM="$(sha256_of "$ABS_PATH")"
    echo "  sha256: $CHECKSUM"
fi

# ---------------------------------------------------------------------------
# Manifest must already exist and contain this entry
# ---------------------------------------------------------------------------

if [[ ! -f "$MANIFEST" ]]; then
    echo "Error: Manifest not found at $MANIFEST" >&2
    echo "       Create it manually with a header row and at least one entry." >&2
    exit 1
fi

NEW_ROW="$REL_PATH,$CHECKSUM,$SOURCE,$SOURCE_TYPE"

ESCAPED_PATH="${REL_PATH//\//\\/}"   # escape slashes for sed

if ! grep -q "^${ESCAPED_PATH}," "$MANIFEST" 2>/dev/null; then
    echo "Error: '$REL_PATH' is not in the manifest." >&2
    echo "       New entries must be added to manifest.csv by hand." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Update the existing row in-place
# ---------------------------------------------------------------------------

if sed --version &>/dev/null 2>&1; then
    # GNU sed
    sed -i "s|^${ESCAPED_PATH},.*|${NEW_ROW}|" "$MANIFEST"
else
    # BSD sed (macOS)
    sed -i '' "s|^${ESCAPED_PATH},.*|${NEW_ROW}|" "$MANIFEST"
fi

echo "Updated: $REL_PATH"
echo "Done. Manifest: $MANIFEST"
