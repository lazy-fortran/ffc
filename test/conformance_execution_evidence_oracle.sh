#!/usr/bin/env bash
# Behavioral oracle for schema-2 compile/run termination evidence.

set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
scratch_base=/mnt/storage
if [ ! -d "$scratch_base" ] || [ ! -w "$scratch_base" ]; then
    scratch_base=${TMPDIR:-/tmp}
fi
scratch=$(mktemp -d "$scratch_base/ffc_execution_evidence_XXXXXX")
trap 'rm -rf "$scratch"' EXIT

fortfront="$scratch/fortfront"
suite_root="$fortfront/examples/lf"
compiler="$scratch/ffc-fixture"
observations="$scratch/observations.jsonl"
report="$scratch/report.jsonl"
log="$scratch/gauntlet.log"
empty_manifest="$scratch/empty.txt"
mkdir -p "$suite_root"
: > "$empty_manifest"

for fixture in compile_failure compile_timeout compile_signal runtime_failure \
        runtime_timeout runtime_signal runtime_exit_124 runtime_exit_137; do
    printf 'print 1\n' > "$suite_root/$fixture.lf"
done

printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'source_path=$1' \
    'output=' \
    'while [ "$#" -gt 0 ]; do' \
    '    if [ "$1" = -o ]; then output=$2; shift 2; else shift; fi' \
    'done' \
    'case "$(basename "$source_path")" in' \
    '    compile_failure.lf) exit 7 ;;' \
    '    compile_timeout.lf) sleep 10; exit 0 ;;' \
    '    compile_signal.lf) kill -SEGV $$ ;;' \
    '    runtime_failure.lf) body="exit 23" ;;' \
    '    runtime_timeout.lf) body="sleep 10" ;;' \
    '    runtime_signal.lf) body="kill -SEGV \$\$" ;;' \
    '    runtime_exit_124.lf) body="exit 124" ;;' \
    '    runtime_exit_137.lf) body="exit 137" ;;' \
    '    *) exit 8 ;;' \
    'esac' \
    'printf "#!/usr/bin/env bash\n%s\n" "$body" > "$output"' \
    'chmod +x "$output"' > "$compiler"
chmod +x "$compiler"

status=0
FFC_FORTFRONT_DIR="$fortfront" \
FFC_XFAIL_MANIFEST="$empty_manifest" \
FFC_SKIP_MANIFEST="$empty_manifest" \
FFC_NOREF_MANIFEST="$empty_manifest" \
bash "$project_dir/scripts/conformance_gauntlet.sh" \
    --suite fortfront-lf \
    --file compile_failure.lf \
    --file compile_timeout.lf \
    --file compile_signal.lf \
    --file runtime_failure.lf \
    --file runtime_timeout.lf \
    --file runtime_signal.lf \
    --file runtime_exit_124.lf \
    --file runtime_exit_137.lf \
    --ffc "$compiler" --timeout 1 \
    --observations "$observations" --report "$report" > "$log" 2>&1 || status=$?
[ "$status" -eq 1 ] || {
    printf 'gauntlet returned %s, expected classified failures\n' "$status" >&2
    cat "$log" >&2
    exit 1
}

python3 - "$observations" "$project_dir/scripts/conformance_observation.py" <<'PY'
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

path = Path(sys.argv[1])
validator = Path(sys.argv[2])
records = [json.loads(line) for line in path.read_text().splitlines()]
summary = records[-1]
rows = {record["file"]: record for record in records[:-1]}

assert set(rows) == {
    "compile_failure.lf",
    "compile_timeout.lf",
    "compile_signal.lf",
    "runtime_failure.lf",
    "runtime_timeout.lf",
    "runtime_signal.lf",
    "runtime_exit_124.lf",
    "runtime_exit_137.lf",
}
epoch = summary["epoch_sha256"]
assert len(epoch) == 64
assert all(row["epoch_sha256"] == epoch for row in rows.values())

compile_failure = rows["compile_failure.lf"]
assert compile_failure["action"] == "compile-run"
assert (
    compile_failure["ffc_compile_action"],
    compile_failure["ffc_compile_exit"],
    compile_failure["ffc_compile_termination"],
    compile_failure["ffc_compile_signal"],
) == ("executed", 7, "exit", 0)
assert (
    compile_failure["ffc_run_action"],
    compile_failure["ffc_run_exit"],
    compile_failure["ffc_run_termination"],
    compile_failure["ffc_run_signal"],
) == ("not-run", -1, "not-run", 0)
assert compile_failure["ffc_exit"] == 7

compile_timeout = rows["compile_timeout.lf"]
assert (
    compile_timeout["ffc_compile_exit"],
    compile_timeout["ffc_compile_termination"],
    compile_timeout["ffc_compile_signal"],
) == (124, "timeout", 15)
assert compile_timeout["ffc_run_action"] == "not-run"

compile_signal = rows["compile_signal.lf"]
assert (
    compile_signal["ffc_compile_exit"],
    compile_signal["ffc_compile_termination"],
    compile_signal["ffc_compile_signal"],
) == (139, "signal", 11)
assert compile_signal["ffc_run_action"] == "not-run"

runtime_failure = rows["runtime_failure.lf"]
assert runtime_failure["ffc_compile_exit"] == 0
assert (
    runtime_failure["ffc_run_exit"],
    runtime_failure["ffc_run_termination"],
    runtime_failure["ffc_run_signal"],
) == (23, "exit", 0)
assert runtime_failure["ffc_exit"] == 23

runtime_timeout = rows["runtime_timeout.lf"]
assert (
    runtime_timeout["ffc_run_exit"],
    runtime_timeout["ffc_run_termination"],
    runtime_timeout["ffc_run_signal"],
) == (124, "timeout", 15)

runtime_signal = rows["runtime_signal.lf"]
assert (
    runtime_signal["ffc_run_exit"],
    runtime_signal["ffc_run_termination"],
    runtime_signal["ffc_run_signal"],
) == (139, "signal", 11)

runtime_exit_124 = rows["runtime_exit_124.lf"]
assert (
    runtime_exit_124["ffc_run_exit"],
    runtime_exit_124["ffc_run_termination"],
    runtime_exit_124["ffc_run_signal"],
) == (124, "exit", 0)

runtime_exit_137 = rows["runtime_exit_137.lf"]
assert (
    runtime_exit_137["ffc_run_exit"],
    runtime_exit_137["ffc_run_termination"],
    runtime_exit_137["ffc_run_signal"],
) == (137, "exit", 0)

for row in rows.values():
    assert row["ref_compile_action"] == "not-run"
    assert row["ref_compile_exit"] == -1
    assert row["ref_run_action"] == "not-run"
    assert row["ref_run_exit"] == -1

def validator_rejects(mutated, expected):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as stream:
        for record in mutated:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        stream.flush()
        result = subprocess.run(
            [sys.executable, str(validator), "validate", "--suite",
             "fortfront-lf", stream.name],
            text=True, capture_output=True,
        )
    assert result.returncode != 0
    assert expected in result.stderr

wrong_epoch = copy.deepcopy(records)
wrong_epoch[1]["epoch_sha256"] = "0" * 64
validator_rejects(wrong_epoch, "differs from SUMMARY: epoch_sha256")

wrong_termination = copy.deepcopy(records)
timeout_index = next(
    index for index, record in enumerate(wrong_termination)
    if record.get("file") == "runtime_timeout.lf"
)
wrong_termination[timeout_index]["ffc_run_termination"] = "exit"
validator_rejects(wrong_termination, "inconsistent termination evidence")
PY
