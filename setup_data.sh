#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# setup_data.sh
#
# Creates the project data directory, verifies existing files against the
# manifest, and fetches any missing or corrupt files.
#
# Usage:
#   ./setup_data.sh [DATA_ROOT]
#
# Arguments:
#   DATA_ROOT   Optional. Absolute path to the shared data root directory.
#               Overrides the SHARED_DATA_ROOT environment variable.
#               Defaults to <project_root>/../data/
#
# Environment variables:
#   SHARED_DATA_ROOT        Path to the data root directory. Set this in
#                           a local environment variables file or in your 
#                           shell profile to avoid passing the argument
#                           each time.
# -----------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
DEFAULT_DATA_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)/data"

# Priority: CLI argument > env var > default
if [[ $# -ge 1 ]]; then
    DATA_ROOT="$(realpath -m "$1")"
elif [[ -n "${SHARED_DATA_ROOT:-}" ]]; then
    DATA_ROOT="$(realpath -m "$SHARED_DATA_ROOT")"
else
    DATA_ROOT="$DEFAULT_DATA_ROOT"
fi

MANIFEST="$PROJECT_ROOT/manifest.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; NC='\033[0m'

info()    { echo -e "  ${GREEN}✔${NC}  $*"; }
warn()    { echo -e "  ${YELLOW}!${NC}  $*"; }
error()   { echo -e "  ${RED}✘${NC}  $*" >&2; }
section() { echo -e "\n$*"; }

sha256_of() {
    if command -v sha256sum &>/dev/null; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'   # macOS fallback
    fi
}

fetch_file() {
    local source="$1" dest="$2" source_type="$3"
    mkdir -p "$(dirname "$dest")"
    case "$source_type" in
        local_copy)
            if [[ ! -e "$source" ]]; then
                error "Local source not found: $source"
                return 1
            fi
            cp "$source" "$dest"
            ;;
        local_symlink)
            if [[ ! -e "$source" ]]; then
                error "Symlink source not found: $source"
                return 1
            fi
            ln -sf "$source" "$dest"
            ;;
        download)
            if command -v curl &>/dev/null; then
                curl -fsSL "$source" -o "$dest"
            elif command -v wget &>/dev/null; then
                wget -q "$source" -O "$dest"
            else
                error "Neither curl nor wget found. Cannot download: $source"
                return 1
            fi
            ;;
        *)
            error "Unknown source_type '$source_type' for: $dest"
            return 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

if [[ ! -f "$MANIFEST" ]]; then
    error "Manifest not found at $MANIFEST"
    exit 1
fi

if ! command -v awk &>/dev/null; then
    error "awk is required but not found."
    exit 1
fi

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

section "Data root : $DATA_ROOT"
section "Manifest  : $MANIFEST"

mkdir -p "$DATA_ROOT"

# Track outcomes
n_ok=0; n_fetched=0; n_failed=0

# Read manifest (skip header and blank/comment lines)
while IFS=',' read -r rel_path expected_sha256 source source_type; do
    # Strip UTF-8 BOM if present (added by Excel and some editors)
    rel_path="${rel_path#$'\xEF\xBB\xBF'}"

    # Skip header row and comments
    [[ "$rel_path" == "path" || "$rel_path" == \#* || -z "$rel_path" ]] && continue

    # Strip any surrounding whitespace or carriage returns
    rel_path="${rel_path//[$'\r\n ']/}"
    expected_sha256="${expected_sha256//[$'\r\n ']/}"
    source="${source//[$'\r\n']/}"
    source_type="${source_type//[$'\r\n ']/}"

    dest="$DATA_ROOT/$rel_path"

    needs_fetch=false

    if [[ -L "$dest" ]]; then
        # It's a symlink — just verify the target exists; skip checksum
        if [[ ! -e "$dest" ]]; then
            warn "Dangling symlink, re-fetching: $rel_path"
            rm -f "$dest"
            needs_fetch=true
        else
            info "OK (symlink): $rel_path"
            (( n_ok++ )) || true
        fi
    elif [[ -f "$dest" ]]; then
        actual_sha256="$(sha256_of "$dest")"
        if [[ "$actual_sha256" == "$expected_sha256" ]]; then
            info "OK: $rel_path"
            (( n_ok++ )) || true
        else
            warn "Checksum mismatch, re-fetching: $rel_path"
            warn "  expected: $expected_sha256"
            warn "  actual:   $actual_sha256"
            needs_fetch=true
        fi
    else
        echo "  …  Missing, fetching: $rel_path"
        needs_fetch=true
    fi

    if [[ "$needs_fetch" == true ]]; then
        if fetch_file "$source" "$dest" "$source_type"; then
            # Verify after fetch (skip for symlinks)
            if [[ "$source_type" != "local_symlink" ]]; then
                actual_sha256="$(sha256_of "$dest")"
                if [[ "$actual_sha256" == "$expected_sha256" ]]; then
                    info "Fetched and verified: $rel_path"
                    (( n_fetched++ )) || true
                else
                    error "Post-fetch checksum mismatch: $rel_path"
                    error "  expected: $expected_sha256"
                    error "  actual:   $actual_sha256"
                    (( n_failed++ )) || true
                fi
            else
                info "Fetched (symlink): $rel_path"
                (( n_fetched++ )) || true
            fi
        else
            (( n_failed++ )) || true
        fi
    fi

done < "$MANIFEST"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "────────────────────────────────────"
echo "  OK: $n_ok   Fetched: $n_fetched   Failed: $n_failed"
echo "────────────────────────────────────"

if [[ $n_failed -gt 0 ]]; then
    error "$n_failed file(s) could not be fetched or verified. See above."
    exit 1
fi
