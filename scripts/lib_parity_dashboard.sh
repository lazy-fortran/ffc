#!/usr/bin/env bash

set -uo pipefail

git_revision() {
    git -C "$1" rev-parse HEAD 2>/dev/null || printf '%040d\n' 0
}

snapshot_field() {
    local kind="$1" name="$2" path="$3"
    awk -F '\t' -v kind="$kind" -v name="$name" \
        '$1 == kind && $2 == name { print $3 }' "$path"
}

validate_snapshot_freshness() {
    local path="$1" ffc_revision recorded_source
    ffc_revision=$(snapshot_field revision ffc "$path")
    recorded_source=$(snapshot_field digest ffc-source "$path")
    git -C "$PROJECT_DIR" merge-base --is-ancestor "$ffc_revision" HEAD \
        2>/dev/null || fail "snapshot ffc revision is not an ancestor"
    [ "$(ffc_revision_source_sha256 "$PROJECT_DIR" "$ffc_revision")" = \
        "$recorded_source" ] || fail "snapshot ffc revision source mismatch"
    [ "$(snapshot_field digest manifests "$path")" = \
        "$EXPECTED_MANIFEST_SHA256" ] || fail "stale snapshot manifest digest"
    [ "$(snapshot_field revision FortFront "$path")" = \
        "$EXPECTED_FORTFRONT_REVISION" ] || fail "stale snapshot FortFront revision"
    [ "$(snapshot_field revision LIRIC "$path")" = \
        "$EXPECTED_LIRIC_REVISION" ] || fail "stale snapshot LIRIC revision"
    [ "$(snapshot_field revision LFortran "$path")" = \
        "$EXPECTED_LFORTRAN_REVISION" ] || fail "stale snapshot LFortran revision"
    [ "$(snapshot_field revision GCC "$path")" = \
        "$EXPECTED_GCC_REVISION" ] || fail "stale snapshot GCC revision"
    [ "$(snapshot_field tree FortFront "$path")" = \
        "$EXPECTED_FORTFRONT_TREE" ] || fail "stale snapshot FortFront tree"
    [ "$(snapshot_field tree LIRIC "$path")" = \
        "$EXPECTED_LIRIC_TREE" ] || fail "stale snapshot LIRIC tree"
    [ "$(snapshot_field corpus-files fortfront-f90 "$path")" = \
        "$EXPECTED_FORTFRONT_F90_FILES" ] || fail "stale snapshot fortfront-f90 files"
    [ "$(snapshot_field corpus-files fortfront-lf "$path")" = \
        "$EXPECTED_FORTFRONT_LF_FILES" ] || fail "stale snapshot fortfront-lf files"
    if [ -d "$(suite_root lfortran)" ]; then
        [ "$(snapshot_field tree LFortran "$path")" = \
            "$EXPECTED_LFORTRAN_TREE" ] || fail "stale snapshot LFortran tree"
        [ "$(snapshot_field corpus-files lfortran "$path")" = \
            "$EXPECTED_LFORTRAN_FILES" ] || fail "stale snapshot LFortran files"
    fi
    if [ -d "$(suite_root gfortran-dg)" ]; then
        [ "$(snapshot_field tree GCC "$path")" = \
            "$EXPECTED_GCC_TREE" ] || fail "stale snapshot GCC tree"
        [ "$(snapshot_field corpus-files gfortran-dg "$path")" = \
            "$EXPECTED_GFORTRAN_DG_FILES" ] || fail "stale snapshot GCC files"
    fi
}

suite_root() {
    case "$1" in
        fortfront-f90) printf '%s\n' "${FFC_FORTFRONT_DIR:-$CORPUS_PARENT/fortfront}/examples/f90" ;;
        fortfront-lf) printf '%s\n' "${FFC_FORTFRONT_DIR:-$CORPUS_PARENT/fortfront}/examples/lf" ;;
        lfortran) printf '%s\n' "${FFC_LFORTRAN_DIR:-$CORPUS_PARENT/lfortran}/integration_tests" ;;
        gfortran-dg) printf '%s\n' "${FFC_GFORTRAN_DG_DIR:-$CORPUS_PARENT/gcc/gcc/testsuite/gfortran.dg}" ;;
    esac
}

suite_files_sha256() {
    local suite="$1" root
    root=$(suite_root "$suite")
    [ -d "$root" ] || return 1
    case "$suite" in
        fortfront-lf)
            find "$root" -maxdepth 1 \( -name '*.lf' -o -name '*.f90' \) -type f ;;
        *) find "$root" -maxdepth 1 -name '*.f90' -type f ;;
    esac | sed "s#^$root/##" | LC_ALL=C sort | sha256sum | cut -d ' ' -f 1
}

rate() {
    local numerator="$1" denominator="$2"
    if [ "$denominator" -eq 0 ]; then
        printf '%s' N/A
    else
        awk -v numerator="$numerator" -v denominator="$denominator" \
            'BEGIN { printf "%.1f%%", 100 * numerator / denominator }'
    fi
}

render_identity() {
    local identity="$1" owner repository issue_number
    case "$identity" in
        owner=*)
            owner=${identity#owner=}
            repository=${owner%#*}
            issue_number=${owner##*#}
            printf '[%s](https://github.com/%s/issues/%s)' \
                "$owner" "$repository" "$issue_number"
            ;;
        scope=*) printf '`scope:%s`' "${identity#scope=}" ;;
        *) printf '%s' "$identity" ;;
    esac
}

validate_snapshot() {
    local path="$1"
    awk -F '\t' -v expected_scope_count="$EXPECTED_SCOPE_COUNT" '
        function invalid(message) {
            printf "ERROR: %s:%d: %s\n", FILENAME, FNR, message > "/dev/stderr"
            failed = 1
        }
        function counter(value) { return value ~ /^[0-9]+$/ }
        $1 == "schema" {
            if (NF != 2 || $2 != 1 || schema_seen++) invalid("invalid schema")
            next
        }
        $1 == "revision" {
            if (NF != 3 || $2 !~ /^(ffc|FortFront|LIRIC|LFortran|GCC)$/ ||
                    length($3) != 40 || $3 !~ /^[0-9A-Fa-f]+$/ ||
                    $3 ~ /^0+$/ || revision[$2]++) invalid("invalid revision")
            next
        }
        $1 == "tree" {
            if (NF != 3 || $2 !~ /^(FortFront|LIRIC|LFortran|GCC)$/ ||
                    length($3) != 40 || $3 !~ /^[0-9A-Fa-f]+$/ ||
                    tree[$2]++) invalid("invalid tree")
            next
        }
        $1 == "corpus-files" {
            if (NF != 3 || $2 !~ /^(fortfront-f90|fortfront-lf|lfortran|gfortran-dg)$/ ||
                    length($3) != 64 || $3 !~ /^[0-9A-Fa-f]+$/ ||
                    corpus_files[$2]++) invalid("invalid corpus file digest")
            next
        }
        $1 == "digest" {
            if (NF != 3 || $2 !~ /^(ffc-(source|binary)|manifests)$/ ||
                    length($3) != 64 || $3 !~ /^[0-9A-Fa-f]+$/ ||
                    digest[$2]++) invalid("invalid digest")
            next
        }
        $1 == "suite" || $1 == "view" || $1 == "excluded" {
            if (NF != 9) { invalid("invalid totals row"); next }
            if (($1 == "suite" && $2 !~ /^(fortfront-f90|fortfront-lf|lfortran|gfortran-dg)$/) ||
                    ($1 == "view" && $2 !~ /^(All|Scoped)$/) ||
                    ($1 == "excluded" && $2 != "Scoped")) {
                invalid("invalid totals label")
            }
            for (i = 3; i <= 9; i++) {
                if (!counter($i)) invalid("invalid counter")
            }
            if ($3 != $4 + $5 + $6 + $7 + $9) invalid("inconsistent total")
            if ($8 > $4 + $6) invalid("inconsistent NOREF count")
            key = $1 SUBSEP $2
            if (total_seen[key]++) invalid("duplicate totals row")
            for (i = 3; i <= 9; i++) {
                if ($1 == "suite") suite_sum[i] += $i
                else if ($1 == "view" && $2 == "All") all_total[i] = $i
                else if ($1 == "view") scoped_total[i] = $i
                else excluded_total[i] = $i
            }
            if ($1 == "suite") {
                suite_status[$2 SUBSEP "XFAIL"] = $5
                suite_status[$2 SUBSEP "XPASS"] = $6
                suite_status[$2 SUBSEP "FAIL"] = $7
                suite_status[$2 SUBSEP "SKIP"] = $9
            }
            next
        }
        $1 == "owner" {
            if (NF != 6 || $2 !~ /^(fortfront-f90|fortfront-lf|lfortran|gfortran-dg)$/ ||
                    $3 !~ /^(XFAIL|XPASS|FAIL|SKIP)$/ ||
                    $4 !~ /^(arrays|characters|control-flow|data|diagnostics|io-runtime|modules-scope|procedures|derived-oop|intrinsics|corpus|backend)$/ ||
                    (($5 !~ /^owner=[^ \/]+\/[^ #]+#[0-9]+$/) &&
                     ($5 !~ /^scope=(coarray|OpenMP|OpenACC|GPU|vendor|legacy|compiler-flags|harness)$/)) ||
                    !counter($6)) {
                invalid("invalid owner row")
            }
            key = $2 SUBSEP $3 SUBSEP $5
            if (owner_seen[key]++) invalid("duplicate owner row")
            owner_sum[$2 SUBSEP $3] += $6
            next
        }
        { invalid("unknown snapshot row") }
        END {
            if (!schema_seen) invalid("missing schema")
            split("ffc FortFront LIRIC LFortran GCC", required, " ")
            for (i in required) if (!revision[required[i]]) invalid("missing revision")
            split("FortFront LIRIC LFortran GCC", required_trees, " ")
            for (i in required_trees) if (!tree[required_trees[i]]) invalid("missing tree")
            if (!digest["ffc-source"] || !digest["ffc-binary"] ||
                    !digest["manifests"]) invalid("missing digest")
            split("fortfront-f90 fortfront-lf lfortran gfortran-dg", suites, " ")
            for (i in suites) if (!corpus_files[suites[i]]) invalid("missing corpus file digest")
            for (i in suites) if (!total_seen["suite" SUBSEP suites[i]]) invalid("missing suite")
            if (!total_seen["view" SUBSEP "All"] ||
                    !total_seen["view" SUBSEP "Scoped"]) invalid("missing view")
            if (!total_seen["excluded" SUBSEP "Scoped"]) invalid("missing excluded totals")
            if (excluded_total[3] != expected_scope_count) {
                invalid("excluded total does not equal scope registry")
            }
            for (i = 3; i <= 9; i++) {
                if (all_total[i] != suite_sum[i]) invalid("All totals do not equal suites")
                if (scoped_total[i] > all_total[i]) invalid("Scoped total exceeds All")
                if (scoped_total[i] != all_total[i] - excluded_total[i]) {
                    invalid("Scoped totals do not equal All minus excluded")
                }
            }
            for (key in suite_status) {
                if (owner_sum[key] != suite_status[key]) {
                    invalid("owner totals do not equal suite status")
                }
            }
            if (failed) exit 1
        }
    ' "$path"
}

subsystem_for_identity() {
    case "$1" in
        scope=*) printf '%s' corpus ;;
        owner=*) printf '%s' "${SUBSYSTEM_FOR_OWNER[$1]}" ;;
        *) printf '%s' corpus ;;
    esac
}

emit_view_row() {
    local label="$1" excluded="$2" values total pass xfail xpass fail_count
    local noref skip
    values=$(awk -F '\t' -v excluded="$excluded" '
        function is_excluded(scope) {
            return scope == "scope=coarray" || scope == "scope=OpenMP" ||
                scope == "scope=OpenACC" || scope == "scope=GPU"
        }
        excluded && is_excluded($8) { next }
        {
            total++
            count[$3]++
            noref += $4
        }
        END {
            print total + 0, count["PASS"] + 0, count["XFAIL"] + 0,
                count["XPASS"] + 0, count["FAIL"] + 0, noref + 0,
                count["SKIP"] + 0
        }
    ' OFS=' ' "$JOINED")
    read -r total pass xfail xpass fail_count noref skip <<< "$values"
    printf 'view\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n' \
        "$label" "$total" "$pass" "$xfail" "$xpass" "$fail_count" \
        "$noref" "$skip"
}

emit_excluded_row() {
    local values total pass xfail xpass fail_count noref skip
    values=$(awk -F '\t' '
        function is_excluded(scope) {
            return scope == "scope=coarray" || scope == "scope=OpenMP" ||
                scope == "scope=OpenACC" || scope == "scope=GPU"
        }
        is_excluded($8) {
            total++
            count[$3]++
            noref += $4
        }
        END {
            print total + 0, count["PASS"] + 0, count["XFAIL"] + 0,
                count["XPASS"] + 0, count["FAIL"] + 0, noref + 0,
                count["SKIP"] + 0
        }
    ' OFS=' ' "$JOINED")
    read -r total pass xfail xpass fail_count noref skip <<< "$values"
    printf 'excluded\tScoped\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n' \
        "$total" "$pass" "$xfail" "$xpass" "$fail_count" "$noref" "$skip"
}

emit_suite_row() {
    local suite="$1" values total pass xfail xpass fail_count noref skip
    values=$(awk -F '\t' -v suite="$suite" '
        $1 == suite {
            total++
            count[$3]++
            noref += $4
        }
        END {
            print total + 0, count["PASS"] + 0, count["XFAIL"] + 0,
                count["XPASS"] + 0, count["FAIL"] + 0, noref + 0,
                count["SKIP"] + 0
        }
    ' OFS=' ' "$JOINED")
    read -r total pass xfail xpass fail_count noref skip <<< "$values"
    printf 'suite\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n' \
        "$suite" "$total" "$pass" "$xfail" "$xpass" "$fail_count" \
        "$noref" "$skip"
}

render_snapshot() {
    local snapshot="$1" destination="$2"
    local kind label total pass xfail xpass fail_count noref skip evaluated
    validate_snapshot "$snapshot" || exit 1
    {
        printf '# Parity Status\n\n'
        printf 'Generated by `scripts/generate_parity_dashboard.sh` from the '
        printf 'checked-in parity snapshot.\n\n'
        printf '## Revisions\n\n'
        printf '| Component | Identifier |\n|---|---|\n'
        awk -F '\t' '$1 == "revision" { printf "| %s | `%s` |\n", $2, $3 }' \
            "$snapshot"
        awk -F '\t' '$1 == "digest" { printf "| %s | `%s` |\n", $2, $3 }' \
            "$snapshot"
        awk -F '\t' '$1 == "tree" { printf "| %s tree | `%s` |\n", $2, $3 }' \
            "$snapshot"
        awk -F '\t' '$1 == "corpus-files" { printf "| %s files | `%s` |\n", $2, $3 }' \
            "$snapshot"
        printf '\n## Suite totals\n\n'
        printf '| Suite | Total | PASS | XFAIL | XPASS | FAIL | NOREF | SKIP | '
        printf 'Evaluated pass rate | Strict raw rate |\n'
        printf '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        while IFS=$'\t' read -r kind label total pass xfail xpass fail_count noref skip; do
            [ "$kind" = suite ] || continue
            evaluated=$((total - skip))
            printf '| %s | %d | %d | %d | %d | %d | %d | %d | %s | %s |\n' \
                "$label" "$total" "$pass" "$xfail" "$xpass" "$fail_count" \
                "$noref" "$skip" "$(rate "$pass" "$evaluated")" \
                "$(rate "$pass" "$total")"
        done < "$snapshot"
        printf '\nEvaluated pass rate is `PASS / (TOTAL - SKIP)`. Strict raw rate is '
        printf '`PASS / TOTAL`. NOREF is a subset of PASS or XPASS.\n\n'
        printf '## Scoped totals\n\n'
        printf 'The scoped view excludes entries tagged `coarray`, `OpenMP`, '
        printf '`OpenACC`, or `GPU`; excluded files are not counted as passes.\n\n'
        printf '| View | Total | PASS | XFAIL | XPASS | FAIL | NOREF | SKIP | '
        printf 'Evaluated pass rate | Strict raw rate |\n'
        printf '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        while IFS=$'\t' read -r kind label total pass xfail xpass fail_count noref skip; do
            [ "$kind" = view ] || continue
            evaluated=$((total - skip))
            printf '| %s | %d | %d | %d | %d | %d | %d | %d | %s | %s |\n' \
                "$label" "$total" "$pass" "$xfail" "$xpass" "$fail_count" \
                "$noref" "$skip" "$(rate "$pass" "$evaluated")" \
                "$(rate "$pass" "$total")"
        done < "$snapshot"
        printf '\n## Owned non-passing results\n\n'
        printf '| Suite | Result | Subsystem | Owner or scope | Files |\n'
        printf '|---|---|---|---|---:|\n'
        while IFS=$'\t' read -r kind suite status subsystem identity count; do
            [ "$kind" = owner ] || continue
            printf '| %s | %s | %s | %s | %d |\n' "$suite" "$status" \
                "$subsystem" "$(render_identity "$identity")" "$count"
        done < "$snapshot"
    } > "$destination"
}
