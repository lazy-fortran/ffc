#!/usr/bin/env bash
# lib_conformance.sh: shared helpers for the conformance gauntlet.
# Source this file; do not execute it directly.
#
# Public functions:
#   find_ffc               resolve ffc binary path
#   compile_with_ffc       compile a source file through ffc
#   compile_with_gfortran  compile a source file with gfortran -w
#   run_capture            run an executable with timeout, capture stdout+stderr
#   compare_outputs        compare stdout files and exit statuses

set -uo pipefail

# Resolve the ffc binary. Priority: --ffc arg > FFC_BIN env > build/ tree > PATH.
find_ffc() {
    if [ -n "${FFC_BIN:-}" ]; then
        echo "$FFC_BIN"
        return 0
    fi
    local candidate
    candidate=$(find build -name ffc -type f -executable 2>/dev/null | head -1) || true
    if [ -n "$candidate" ]; then
        echo "$candidate"
        return 0
    fi
    local in_path
    in_path=$(command -v ffc 2>/dev/null) || true
    if [ -n "$in_path" ]; then
        echo "$in_path"
        return 0
    fi
    echo "ERROR: ffc binary not found. Set FFC_BIN or run 'fpm build' first." >&2
    return 1
}

# compile_with_ffc <source> <exe> <ffc_path>
# Returns 0 on success, non-zero on failure.
compile_with_ffc() {
    local source="$1" exe="$2" ffc="$3"
    "$ffc" "$source" -o "$exe" 2>/dev/null
    return $?
}

# compile_with_gfortran <source> <exe>
# Returns 0 on success, non-zero on failure.
compile_with_gfortran() {
    local source="$1" exe="$2"
    gfortran -w "$source" -o "$exe" 2>/dev/null
    return $?
}

# run_capture <exe> <out_file> <timeout_seconds>
# Runs the executable, captures stdout+stderr to out_file, respects timeout.
# Returns the exit status of the executable (or 124 if timed out).
run_capture() {
    local exe="$1" out_file="$2" timeout="$3"
    timeout "$timeout" "$exe" > "$out_file" 2>&1
    return $?
}

# compare_outputs <ffc_out> <ref_out> <ffc_exit> <ref_exit>
# Returns 0 if both stdout and exit codes match, 1 otherwise.
compare_outputs() {
    local ffc_out="$1" ref_out="$2" ffc_exit="$3" ref_exit="$4"
    if [ "$ffc_exit" != "$ref_exit" ]; then
        return 1
    fi
    diff -q "$ffc_out" "$ref_out" > /dev/null 2>&1
    return $?
}

# check_xfail <manifest_path> <suite_relative_path>
# Returns 0 if the path is listed in the xfail manifest, 1 otherwise.
check_xfail() {
    local manifest="$1" path="$2"
    if [ ! -f "$manifest" ]; then
        return 1
    fi
    local stripped
    stripped=$(echo "$path" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    while IFS= read -r line || [ -n "$line" ]; do
        line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [ -z "$line" ] && continue
        case "$line" in \#*) continue ;; esac
        if [ "$line" = "$stripped" ]; then
            return 0
        fi
    done < "$manifest"
    return 1
}

# Escape a string for safe JSON embedding (handles backslash, quotes, newlines).
json_escape() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\r'/\\r}"
    s="${s//$'\t'/\\t}"
    printf '%s' "$s"
}
