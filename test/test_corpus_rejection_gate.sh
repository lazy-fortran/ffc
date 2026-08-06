#!/usr/bin/env bash
# Behavioral oracle for scripts/corpus_rejection_gate.sh.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d /var/tmp/ffc_rejection_gate_test_XXXXXX)
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT
CORPUS="$WORK/corpus"
mkdir -p "$CORPUS"

printf 'program valid\nprint *, 1\nend program valid\n' > "$CORPUS/valid.f90"
printf 'program invalid\n! reject\nend program invalid\n' > "$CORPUS/invalid.f90"

FFC="$WORK/fake-ffc"
cat > "$FFC" <<'EOF'
#!/usr/bin/env bash
set -eu
source=
object=
while (($# > 0)); do
    case "$1" in
        -o) object=$2; shift 2 ;;
        -I) shift 2 ;;
        *.f90|*.F90|*.f|*.F|*.lf) source=$1; shift ;;
        *) shift ;;
    esac
done
if grep -q '! reject' "$source"; then
    echo "fake ffc diagnostic: intentional rejection" >&2
    exit 1
fi
: > "$object"
EOF
chmod +x "$FFC"

GFORTRAN="$WORK/fake-gfortran"
cat > "$GFORTRAN" <<'EOF'
#!/usr/bin/env bash
set -eu
source=
for arg in "$@"; do
    case "$arg" in
        *.f90|*.F90|*.f|*.F|*.lf) source=$arg ;;
    esac
done
if grep -q '! reject' "$source"; then
    echo "fake syntax oracle: invalid source" >&2
    exit 1
fi
exit 0
EOF
chmod +x "$GFORTRAN"

OUT="$WORK/current.tsv"
bash "$ROOT/scripts/corpus_rejection_gate.sh" --ffc "$FFC" \
    --gfortran "$GFORTRAN" --corpus "$CORPUS" --jobs 2 --timeout 5 \
    --out "$OUT"
grep -Fq $'ACCEPTED\t' "$OUT"
grep -Fq $'REJECTED\t' "$OUT"
grep -Fq 'fake ffc diagnostic: intentional rejection' "$OUT.stderr.log"

# Pretend the old compiler accepted both rows.  The new rejection must fail,
# and the independent syntax oracle must classify it as INVALID.
BASELINE="$WORK/baseline.tsv"
sed 's/^REJECTED/ACCEPTED/' "$OUT" > "$BASELINE"
if bash "$ROOT/scripts/corpus_rejection_gate.sh" --ffc "$FFC" \
    --gfortran "$GFORTRAN" --corpus "$CORPUS" --jobs 2 --timeout 5 \
    --out "$WORK/compared.tsv" --baseline "$BASELINE"; then
    echo "expected a newly rejected row" >&2
    exit 1
fi
grep -Fq $'INVALID\t' "$WORK/compared.tsv.validity.tsv"
grep -Fq 'fake syntax oracle: invalid source' \
    "$WORK/compared.tsv.validity.tsv.stderr.log"

# An explicit review allowlist is the only way to make the comparison green.
REJECTED_PATH=$(awk -F '\t' '$1 == "REJECTED" {print $2; exit}' "$OUT")
printf '%s\n' "$REJECTED_PATH" > "$WORK/allow.txt"
bash "$ROOT/scripts/corpus_rejection_gate.sh" --ffc "$FFC" \
    --gfortran "$GFORTRAN" --corpus "$CORPUS" --jobs 2 --timeout 5 \
    --out "$WORK/allowed.tsv" --baseline "$BASELINE" --allow "$WORK/allow.txt"
grep -Fq $'REJECTED\t' "$WORK/allowed.tsv"

echo "PASS: corpus rejection gate baseline, stderr, oracle, and allowlist behavior"

