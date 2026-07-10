#!/usr/bin/env bash

set -uo pipefail

trim_manifest_field() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s\n' "$value"
}

manifest_error() {
    local source="$1" line_number="$2" message="$3"
    printf 'ERROR: %s:%s: %s\n' "$source" "$line_number" "$message" >&2
    return 1
}

validate_manifest_owner() {
    local source="$1" line_number="$2" owner="$3"
    if [[ ! "$owner" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#[1-9][0-9]*$ ]]; then
        manifest_error "$source" "$line_number" "malformed owner: $owner"
    fi
}

validate_manifest_scope() {
    local source="$1" line_number="$2" scope="$3"
    case "$scope" in
        coarray|OpenMP|OpenACC|GPU|vendor|legacy|compiler-flags|harness) ;;
        *) manifest_error "$source" "$line_number" "unknown scope: $scope" ;;
    esac
}

validate_manifest_metadata() {
    local source="$1" line_number="$2" metadata="$3"
    local identity reason
    if [[ "$metadata" != *'; reason='* ]]; then
        if [[ "$metadata" == owner=* || "$metadata" == scope=* ]]; then
            manifest_error "$source" "$line_number" "reason is required"
        else
            manifest_error "$source" "$line_number" \
                "owner or scope is required"
        fi
        return
    fi
    identity=${metadata%%; reason=*}
    reason=${metadata#*; reason=}
    if [[ "$identity" == owner=*scope=* || "$identity" == scope=*owner=* ]]; then
        manifest_error "$source" "$line_number" \
            "owner and scope cannot both appear"
        return
    fi
    reason=$(trim_manifest_field "$reason")
    if [ -z "$reason" ]; then
        manifest_error "$source" "$line_number" "reason is required"
        return
    fi
    case "$identity" in
        owner=*) validate_manifest_owner "$source" "$line_number" \
            "${identity#owner=}" ;;
        scope=*) validate_manifest_scope "$source" "$line_number" \
            "${identity#scope=}" ;;
        *) manifest_error "$source" "$line_number" \
            "owner or scope is required" ;;
    esac
}

validate_manifest_path() {
    local source="$1" line_number="$2" path="$3"
    case "$path" in
        "") manifest_error "$source" "$line_number" "path is required" ;;
        /*) manifest_error "$source" "$line_number" \
            "path must be suite-relative: $path" ;;
        ..|../*|*/../*|*/..) manifest_error "$source" "$line_number" \
            "path contains parent traversal: $path" ;;
    esac
}

validate_expected_manifest() {
    local source="$1" lookup="$2" line line_number=0 path metadata
    declare -A seen_paths=()
    : > "$lookup"
    [ -f "$source" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        line_number=$((line_number + 1))
        line=$(trim_manifest_field "$line")
        [ -z "$line" ] && continue
        [[ "$line" == \#* ]] && continue
        if [[ "$line" != *' # '* ]]; then
            manifest_error "$source" "$line_number" "malformed delimiter"
            return 1
        fi
        path=$(trim_manifest_field "${line%% # *}")
        metadata=${line#* # }
        validate_manifest_path "$source" "$line_number" "$path" || return 1
        validate_manifest_metadata "$source" "$line_number" "$metadata" || \
            return 1
        if [[ -v "seen_paths[$path]" ]]; then
            manifest_error "$source" "$line_number" "duplicate path: $path"
            return 1
        fi
        seen_paths["$path"]=1
        printf '%s\n' "$path" >> "$lookup"
    done < "$source"
}

manifest_owner_references() {
    sed -n 's/^[^#]* # owner=\([^;]*\); reason=.*/\1/p' "$1"
}
