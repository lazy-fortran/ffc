#!/usr/bin/env bash
# Independent standard-Fortran references and optional input for corpus cases.
# A reference is named CASE.ref and input CASE.stdin, retaining CASE's extension.

initialize_case_oracle() {
    local directory="$PROJECT_DIR/test/conformance/oracles/$SUITE"
    local reference="$directory/$rel_path.ref"
    local input="$directory/$rel_path.stdin"
    local dependencies="$CASE_SNAPSHOT_DIR/oracle-dependencies.tsv"
    local status="$CASE_SNAPSHOT_DIR/oracle-status.txt"

    CASE_REFERENCE_SOURCE=""
    CASE_STDIN_FILE="/dev/null"
    CASE_STDIN_SHA256="$EMPTY_SHA256"
    [ -d "$directory" ] || return 0
    mkdir -p "$CASE_SNAPSHOT_DIR"
    : > "$dependencies"
    : > "$status"

    if [ -f "$reference" ]; then
        CASE_REFERENCE_SOURCE=$(python3 \
            "$SCRIPT_DIR/conformance_source_snapshot.py" \
            --suite-root "$directory" --destination "$CASE_SNAPSHOT_DIR/oracles" \
            --manifest "$dependencies" --status "$status" "$reference") || \
            fail "cannot snapshot reference: $rel_path"
        CASE_REF_FLAGS="-w -J @private-module-dir"
    fi
    if [ -f "$input" ]; then
        CASE_STDIN_FILE=$(python3 \
            "$SCRIPT_DIR/conformance_source_snapshot.py" \
            --suite-root "$directory" --destination "$CASE_SNAPSHOT_DIR/oracles" \
            --manifest "$dependencies" --status "$status" "$input") || \
            fail "cannot snapshot stdin: $rel_path"
        CASE_STDIN_SHA256=$(sha256_file_or_empty "$CASE_STDIN_FILE")
    fi
    sed 's/^suite:/oracle:/' "$dependencies" >> "$CASE_DEPENDENCY_FILE"
    cat "$status" >> "$CASE_SNAPSHOT_STATUS"
}
