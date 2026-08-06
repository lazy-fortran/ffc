#!/usr/bin/env bash
# Behavioral and resource oracle for scripts/benchmark_large_translation_unit.sh.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SCRATCH_ROOT=${TMPDIR:-/var/tmp/ert}
mkdir -p "$SCRATCH_ROOT"
WORK=$(mktemp -d "$SCRATCH_ROOT/ffc-benchmark-test.XXXXXX")
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

make_repo() {
    local directory=$1
    mkdir -p "$directory/build/fo/bin"
    printf 'name = "benchmark-fixture"\n' > "$directory/fpm.toml"
    printf '#!/usr/bin/env bash\n' > "$directory/build/fo/bin/ffc"
    printf 'exit 0\n' >> "$directory/build/fo/bin/ffc"
    chmod +x "$directory/build/fo/bin/ffc"
    git -C "$directory" init -q
    git -C "$directory" config user.email test@example.invalid
    git -C "$directory" config user.name benchmark-test
    git -C "$directory" add fpm.toml build/fo/bin/ffc
    git -C "$directory" commit -qm fixture
}

mkdir -p "$WORK/fortfront/examples/f90" "$WORK/bin"
git -C "$WORK/fortfront" init -q
git -C "$WORK/fortfront" config user.email test@example.invalid
git -C "$WORK/fortfront" config user.name benchmark-test

cat > "$WORK/fortfront/examples/f90/benchmark_5000_lines.f90" <<'EOF'
program benchmark
    implicit none
    integer :: result
contains
    integer function bench_func_1(a, b)
        integer, intent(in) :: a, b
        bench_func_1 = a + b
    end function bench_func_1
    integer function bench_func_2(a, b)
        integer, intent(in) :: a, b
        bench_func_2 = a * b
    end function bench_func_2
    integer function bench_func_3(a, b)
        integer, intent(in) :: a, b
        bench_func_3 = a - b
    end function bench_func_3
end program benchmark
EOF
while (( $(wc -l < "$WORK/fortfront/examples/f90/benchmark_5000_lines.f90") < 5000 )); do
    printf '    ! padding line\n' >> "$WORK/fortfront/examples/f90/benchmark_5000_lines.f90"
done
git -C "$WORK/fortfront" add examples/f90/benchmark_5000_lines.f90
git -C "$WORK/fortfront" commit -qm fixture

make_repo "$WORK/baseline"
make_repo "$WORK/candidate"

cat > "$WORK/bin/fo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ ${1:-} == exec ]] || { echo "fake fo only supports exec" >&2; exit 2; }
shift
source=
output=
while (($# > 0)); do
    case "$1" in
        *.f90|*.F90) source=$1; shift ;;
        -o) output=$2; shift 2 ;;
        *) shift ;;
    esac
done
[[ -n $source && -n $output ]] || { echo "fake fo missing source/output" >&2; exit 2; }
if [[ $PWD == */candidate ]]; then
    sleep "${FAKE_CANDIDATE_DELAY:-0}"
fi
gfortran -O0 -w "$source" -o "$output"
EOF
chmod +x "$WORK/bin/fo"

export PATH="$WORK/bin:$PATH"
SOURCE="$WORK/fortfront/examples/f90/benchmark_5000_lines.f90"

bash "$ROOT/scripts/benchmark_large_translation_unit.sh" \
    --baseline-dir "$WORK/baseline" --candidate-dir "$WORK/candidate" \
    --source "$SOURCE" --repeats 3 --max-wall-regression-pct 100 \
    --max-rss-regression-pct 100 --report "$WORK/pass.md"
grep -Fq 'status: PASS' "$WORK/pass.md"
grep -Fq 'stdout matched gfortran byte-for-byte' "$WORK/pass.md"

if FAKE_CANDIDATE_DELAY=1 bash "$ROOT/scripts/benchmark_large_translation_unit.sh" \
    --baseline-dir "$WORK/baseline" --candidate-dir "$WORK/candidate" \
    --source "$SOURCE" --repeats 3 --max-wall-regression-pct 10 \
    --max-rss-regression-pct 100 --report "$WORK/fail.md" \
    > "$WORK/fail.stdout" 2> "$WORK/fail.stderr"; then
    echo "expected the candidate wall-time regression to fail" >&2
    exit 1
fi
grep -Fq 'wall-time regression' "$WORK/fail.stderr"
grep -Fq 'status: FAIL' "$WORK/fail.md"

echo "PASS: benchmark output oracle and wall-time regression gate"
