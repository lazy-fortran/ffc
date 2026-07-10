#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FETCH="$PROJECT_DIR/scripts/fetch_corpora.sh"
SOURCE="${FFC_LFORTRAN_DIR:-$(dirname "$PROJECT_DIR")/lfortran}"
PIN=$("$FETCH" --print-pins | sed -n 's/^lfortran_sha=//p')
EXPECTED_URL="https://github.com/lfortran/lfortran"
WORK=$(mktemp -d /tmp/ffc_fetch_test_XXXXXX)
trap 'rm -rf "$WORK"' EXIT

make_checkout() {
    local dir="$1"
    git clone --shared "$SOURCE" "$dir" >/dev/null 2>&1
    git -C "$dir" remote set-url origin "$EXPECTED_URL"
    git -C "$dir" checkout --detach "$PIN" >/dev/null 2>&1
}

create_origin() {
    local repo="$1" bare="$2" source_path="$3"
    git init "$repo" >/dev/null
    git -C "$repo" config user.name test
    git -C "$repo" config user.email test@example.invalid
    mkdir -p "$repo/$(dirname "$source_path")"
    printf 'program smoke\nend program smoke\n' > "$repo/$source_path"
    git -C "$repo" add "$source_path"
    git -C "$repo" commit -m initial >/dev/null
    git clone --bare "$repo" "$bare" >/dev/null 2>&1
    git -C "$bare" config uploadpack.allowFilter true
}

run_test_fetch() {
    FFC_CORPUS_TEST_MODE=1 \
        FFC_LFORTRAN_URL="file://$LF_ORIGIN" \
        FFC_GCC_URL="file://$GCC_ORIGIN" \
        FFC_LFORTRAN_SHA="$LF_TEST_SHA" \
        FFC_GCC_SHA="$GCC_TEST_SHA" \
        FFC_LFORTRAN_DIR="$LF_TARGET" \
        FFC_GFORTRAN_DG_DIR="$GCC_TARGET/gcc/testsuite/gfortran.dg" \
        "$FETCH" "$@"
}

expect_failure() {
    local name="$1" expected="$2" dir="$3" mode="${4:-verify}"
    local output
    if [ "$mode" = "normal" ]; then
        if output=$(FFC_LFORTRAN_DIR="$dir" "$FETCH" lfortran 2>&1); then
            echo "FAIL: $name unexpectedly passed" >&2
            return 1
        fi
    else
        if output=$(FFC_LFORTRAN_DIR="$dir" "$FETCH" --verify-only lfortran 2>&1); then
            echo "FAIL: $name unexpectedly passed" >&2
            return 1
        fi
    fi
    if ! grep -Fq "$expected" <<< "$output"; then
        echo "FAIL: $name missing diagnostic: $expected" >&2
        echo "$output" >&2
        return 1
    fi
}

expect_failure "missing" "checkout missing" "$WORK/missing"

make_checkout "$WORK/wrong-origin"
git -C "$WORK/wrong-origin" remote set-url origin https://example.invalid/lfortran
expect_failure "wrong origin" "origin is" "$WORK/wrong-origin"

make_checkout "$WORK/attached"
git -C "$WORK/attached" switch -c test-attached >/dev/null 2>&1
expect_failure "attached" "must be detached" "$WORK/attached" "normal"

make_checkout "$WORK/dirty"
touch "$WORK/dirty/untracked.marker"
expect_failure "dirty" "checkout is dirty" "$WORK/dirty"

make_checkout "$WORK/wrong-sha"
git -C "$WORK/wrong-sha" -c user.name=test -c user.email=test@example.invalid \
    commit --allow-empty -m drift >/dev/null 2>&1
expect_failure "wrong SHA" "HEAD is" "$WORK/wrong-sha"

make_checkout "$WORK/incomplete"
git -C "$WORK/incomplete" sparse-checkout init --cone
git -C "$WORK/incomplete" sparse-checkout set src
expect_failure "incomplete" "corpus is incomplete" "$WORK/incomplete"

mkdir -p "$WORK/non-git"
touch "$WORK/non-git/marker"
expect_failure "non-git" "not a git checkout" "$WORK/non-git" "normal"

LF_REPO="$WORK/lfortran-repo"
LF_ORIGIN="$WORK/lfortran-origin.git"
GCC_REPO="$WORK/gcc-repo"
GCC_ORIGIN="$WORK/gcc-origin.git"
LF_TARGET="$WORK/lfortran-fresh"
GCC_TARGET="$WORK/gcc-fresh"
create_origin "$LF_REPO" "$LF_ORIGIN" "integration_tests/smoke.f90"
create_origin "$GCC_REPO" "$GCC_ORIGIN" \
    "gcc/testsuite/gfortran.dg/smoke.f90"
LF_TEST_SHA=$(git -C "$LF_REPO" rev-parse HEAD)
GCC_TEST_SHA=$(git -C "$GCC_REPO" rev-parse HEAD)

run_test_fetch >/dev/null
test "$(git -C "$LF_TARGET" rev-parse HEAD)" = "$LF_TEST_SHA"
test "$(git -C "$GCC_TARGET" rev-parse HEAD)" = "$GCC_TEST_SHA"
! git -C "$LF_TARGET" symbolic-ref -q HEAD >/dev/null
! git -C "$GCC_TARGET" symbolic-ref -q HEAD >/dev/null
test "$(git -C "$GCC_TARGET" sparse-checkout list)" = \
    "gcc/testsuite/gfortran.dg"
test "$(git -C "$GCC_TARGET" config --get remote.origin.promisor)" = "true"
test "$(git -C "$GCC_TARGET" config --get remote.origin.partialclonefilter)" = \
    "blob:none"

mv "$LF_ORIGIN" "$LF_ORIGIN.offline"
mv "$GCC_ORIGIN" "$GCC_ORIGIN.offline"
run_test_fetch --verify-only >/dev/null
mv "$LF_ORIGIN.offline" "$LF_ORIGIN"
mv "$GCC_ORIGIN.offline" "$GCC_ORIGIN"

UNREACHABLE_SENTINEL="$WORK/suite-ran"
if FFC_CORPUS_TEST_MODE=1 \
    FFC_LFORTRAN_URL="file://$LF_ORIGIN" \
    FFC_GCC_URL="file://$GCC_ORIGIN" \
    FFC_LFORTRAN_SHA="0000000000000000000000000000000000000000" \
    FFC_GCC_SHA="$GCC_TEST_SHA" \
    FFC_LFORTRAN_DIR="$WORK/unreachable" \
    "$FETCH" lfortran >/dev/null 2>&1 && touch "$UNREACHABLE_SENTINEL"; then
    echo "FAIL: unreachable pin unexpectedly fetched" >&2
    exit 1
fi
test ! -e "$UNREACHABLE_SENTINEL"

FFC_LFORTRAN_DIR="$SOURCE" "$FETCH" --verify-only lfortran >/dev/null
echo "PASS: pinned corpus verifier rejects invalid checkouts"
