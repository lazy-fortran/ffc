#!/usr/bin/env bash
# lib_conformance.sh: shared helpers for the conformance gauntlet.
# Source this file; do not execute it directly.
#
# Public functions:
#   find_ffc               resolve ffc binary path
#   compile_with_ffc       compile a source file through ffc
#   compile_object_with_ffc  compile a source file through ffc -c
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
    # Pick the most recently built ffc. Several may coexist (build/fo/bin/ffc
    # from the fo backend, build/gfortran_*/app/ffc from fpm); an arbitrary
    # head -1 can return a stale one whose lowering predates recent fixes.
    candidate=$(find build -name ffc -type f -executable -printf '%T@ %p\n' \
        2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-) || true
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
    timeout "${FFC_COMPILE_TIMEOUT:-10}" "$ffc" "$source" -o "$exe" 2>/dev/null
    return $?
}

# compile_object_with_ffc <source> <object> <ffc_path>
# Returns 0 on success, non-zero on failure.
compile_object_with_ffc() {
    local source="$1" object="$2" ffc="$3"
    timeout "${FFC_COMPILE_TIMEOUT:-10}" "$ffc" "$source" -c -o "$object" 2>/dev/null
    return $?
}

# compile_with_gfortran <source> <exe>
# Returns 0 on success, non-zero on failure.
compile_with_gfortran() {
    local source="$1" exe="$2"
    timeout "${GFORTRAN_COMPILE_TIMEOUT:-${FFC_COMPILE_TIMEOUT:-10}}" \
        gfortran -w "$source" -o "$exe" 2>/dev/null
    return $?
}

# run_capture <exe> <out_file> <timeout_seconds>
# Runs the executable, captures stdout+stderr to out_file, respects timeout.
# Returns the exit status of the executable (or 124 if timed out).
# stdin is redirected from /dev/null: the caller drives the file list on the
# loop's stdin, and a test program that reads stdin (PAUSE, READ) must not steal
# those lines and corrupt later iterations.
run_capture() {
    local exe="$1" out_file="$2" timeout="$3"
    timeout "$timeout" "$exe" > "$out_file" 2>&1 < /dev/null
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

# numeric_structure <file>
# Emit the file's output with every numeric literal collapsed to a single 'N'
# token and runs of whitespace collapsed to one space. Two outputs share a
# numeric structure when they differ only in the magnitude and field width of
# their numbers, which is the signature of a nondeterministic timing/random
# program.
numeric_structure() {
    local file="$1"
    sed -E 's/[+-]?[0-9]+\.?[0-9]*([eEdD][+-]?[0-9]+)?/N/g; s/[[:space:]]+/ /g; s/^ //; s/ $//' \
        "$file"
}

# reference_is_nondeterministic <exe> <timeout>
# True when two runs of the reference executable produce different stdout but a
# matching numeric structure. Such programs (CPU_TIME, SYSTEM_CLOCK, RANDOM_*)
# cannot be compared byte-for-byte; they are compared structurally instead.
reference_is_nondeterministic() {
    local exe="$1" timeout="$2"
    local a b
    a=$(timeout "$timeout" "$exe" 2>&1 < /dev/null) || return 1
    b=$(timeout "$timeout" "$exe" 2>&1 < /dev/null) || return 1
    [ "$a" != "$b" ]
}

# compare_structural <ffc_out> <ref_out> <ffc_exit> <ref_exit>
# Like compare_outputs but ignores numeric magnitudes and field widths. Used
# only when the reference program is nondeterministic.
compare_structural() {
    local ffc_out="$1" ref_out="$2" ffc_exit="$3" ref_exit="$4"
    if [ "$ffc_exit" != "$ref_exit" ]; then
        return 1
    fi
    [ "$(numeric_structure "$ffc_out")" = "$(numeric_structure "$ref_out")" ]
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
    grep -Fxq -- "$stripped" "$manifest"
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
