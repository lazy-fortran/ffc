#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 --baseline-dir DIR --candidate-dir DIR --source FILE" >&2
    echo "          [--repeats ODD_NUMBER] [--report FILE]" >&2
}

baseline_dir=
candidate_dir=
source_file=
repeats=3
report_file=

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline-dir)
            baseline_dir=$2
            shift 2
            ;;
        --candidate-dir)
            candidate_dir=$2
            shift 2
            ;;
        --source)
            source_file=$2
            shift 2
            ;;
        --repeats)
            repeats=$2
            shift 2
            ;;
        --report)
            report_file=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

if [[ -z $baseline_dir || -z $candidate_dir || -z $source_file ]]; then
    usage
    exit 2
fi
if [[ ! $repeats =~ ^[0-9]+$ ]] || ((repeats < 3 || repeats % 2 == 0)); then
    echo "--repeats must be an odd integer of at least 3" >&2
    exit 2
fi
for directory in "$baseline_dir" "$candidate_dir"; do
    if [[ ! -f $directory/fpm.toml ]]; then
        echo "not an ffc worktree: $directory" >&2
        exit 2
    fi
    if [[ ! -x $directory/build/fo/bin/ffc ]]; then
        echo "missing fo-built ffc executable; run fo build first: $directory" >&2
        exit 2
    fi
done
if [[ -n $(git -C "$baseline_dir" status --porcelain) ]]; then
    echo "baseline worktree must be clean: $baseline_dir" >&2
    exit 2
fi
if [[ ! -f $source_file ]]; then
    echo "source not found: $source_file" >&2
    exit 2
fi
if ! command -v gfortran >/dev/null; then
    echo "gfortran is required for the behavioral oracle" >&2
    exit 2
fi
if [[ ! -x /usr/bin/time ]]; then
    echo "GNU /usr/bin/time is required for resource measurements" >&2
    exit 2
fi

baseline_dir=$(realpath "$baseline_dir")
candidate_dir=$(realpath "$candidate_dir")
source_file=$(realpath "$source_file")
fortfront_dir=$(realpath "$candidate_dir/../fortfront")
if [[ $fortfront_dir != "$(realpath "$baseline_dir/../fortfront")" ]]; then
    echo "baseline and candidate must use the same FortFront worktree" >&2
    exit 2
fi
if [[ -n $(git -C "$fortfront_dir" status --porcelain) ]]; then
    echo "FortFront dependency worktree must be clean: $fortfront_dir" >&2
    exit 2
fi
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/ffc-large-unit.XXXXXX")
trap 'rm -rf -- "$tmp_dir"' EXIT
raw_file=$tmp_dir/raw.tsv
generated_source=$tmp_dir/benchmark_5000_lines_behavior.f90
reference_exe=$tmp_dir/reference
reference_out=$tmp_dir/reference.out

function_count=$(awk '/^[[:space:]]*integer function bench_func_[0-9]+/ { n++ } END { print n + 0 }' "$source_file")
if ((function_count == 0)); then
    echo "source has no bench_func_N procedures: $source_file" >&2
    exit 2
fi

# The FortFront fixture has an empty main program. Replace four padding lines
# with calls to its first, middle, and last procedures plus a print. The timed
# source stays at 5,000 lines and gains a behavioral oracle without adding 332
# unrelated call sites to the lowering benchmark.
awk -v function_count="$function_count" '
    BEGIN {
        injected = 0
        removed = 0
        remove_count = 4
        middle = int((function_count + 1) / 2)
    }
    /^[[:space:]]*contains[[:space:]]*$/ && !injected {
        print "    result = bench_func_1(1, 2)"
        print "    result = result + bench_func_" middle "(2, 3)"
        print "    result = result + bench_func_" function_count "(3, 4)"
        print "    print *, result"
        injected = 1
    }
    /^[[:space:]]*! padding line[[:space:]]*$/ && removed < remove_count {
        removed++
        next
    }
    { print }
    END {
        if (!injected || removed != remove_count) exit 3
    }
' "$source_file" > "$generated_source"

source_lines=$(wc -l < "$source_file")
generated_lines=$(wc -l < "$generated_source")
if [[ $source_lines -ne 5000 ]]; then
    echo "benchmark source must contain exactly 5,000 lines: $source_lines" >&2
    exit 2
fi
if [[ $generated_lines -ne $source_lines ]]; then
    echo "generated source changed line count: $source_lines -> $generated_lines" >&2
    exit 1
fi

gfortran -O0 -w "$generated_source" -o "$reference_exe"
"$reference_exe" > "$reference_out"
printf 'compiler\tround\tposition\treal_s\tuser_s\tsystem_s\tmax_rss_kb\tload1\n' > "$raw_file"

run_one() {
    local label=$1
    local directory=$2
    local round=$3
    local position=$4
    local exe=$tmp_dir/$label-$round
    local output=$tmp_dir/$label-$round.out
    local timing=$tmp_dir/$label-$round.time
    local load1
    load1=$(awk '{ print $1 }' /proc/loadavg)
    (
        cd "$directory"
        /usr/bin/time -f '%e\t%U\t%S\t%M' -o "$timing" \
            fo exec --no-build ffc "$generated_source" -o "$exe" \
                --backend isel
    )
    "$exe" > "$output"
    if ! cmp -s "$reference_out" "$output"; then
        echo "$label output differs from gfortran in round $round" >&2
        diff -u "$reference_out" "$output" >&2 || true
        exit 1
    fi
    if [[ $round != warmup ]]; then
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$label" "$round" "$position" "$(cat "$timing")" "$load1" \
            >> "$raw_file"
    fi
}

# Populate filesystem and dynamic-library caches for both compilers before the
# measured, alternating sequence. These runs still execute the output oracle.
run_one baseline "$baseline_dir" warmup 1
run_one candidate "$candidate_dir" warmup 2

for ((round = 1; round <= repeats; round++)); do
    if ((round % 2 == 1)); then
        run_one baseline "$baseline_dir" "$round" 1
        run_one candidate "$candidate_dir" "$round" 2
    else
        run_one candidate "$candidate_dir" "$round" 1
        run_one baseline "$baseline_dir" "$round" 2
    fi
done

median_column() {
    local label=$1
    local column=$2
    awk -F '\t' -v label="$label" -v column="$column" \
        '$1 == label { print $column }' "$raw_file" | sort -n | \
        awk -v middle="$((repeats / 2 + 1))" 'NR == middle { print; exit }'
}

baseline_time=$(median_column baseline 4)
candidate_time=$(median_column candidate 4)
baseline_rss=$(median_column baseline 7)
candidate_rss=$(median_column candidate 7)
speedup=$(awk -v before="$baseline_time" -v after="$candidate_time" \
    'BEGIN { if (after == 0) print "inf"; else printf "%.2f", before / after }')
rss_change=$(awk -v before="$baseline_rss" -v after="$candidate_rss" \
    'BEGIN { if (before == 0) print "nan"; else printf "%.1f", 100 * (after - before) / before }')

gfortran_path=$(realpath "$(command -v gfortran)")
gfortran_version=$(gfortran -dumpfullversion -dumpversion)
gfortran_sha=$(sha256sum "$gfortran_path" | awk '{ print $1 }')
baseline_revision=$(git -C "$baseline_dir" rev-parse HEAD)
candidate_revision=$(git -C "$candidate_dir" rev-parse HEAD)
fortfront_revision=$(git -C "$fortfront_dir" rev-parse HEAD)
fortfront_tree=$(git -C "$fortfront_dir" rev-parse 'HEAD^{tree}')
source_sha=$(sha256sum "$generated_source" | awk '{ print $1 }')
cpu_model=$(awk -F ': ' '/model name/ { print $2; exit }' /proc/cpuinfo)

worktree_content_sha() {
    local directory=$1
    (
        cd "$directory"
        git ls-files --cached --others --exclude-standard -z | sort -z | \
            while IFS= read -r -d '' path; do
                printf '%s\0%s\0' "$(git hash-object -- "$path")" "$path"
            done
    ) | sha256sum | awk '{ print $1 }'
}

baseline_content_sha=$(worktree_content_sha "$baseline_dir")
candidate_content_sha=$(worktree_content_sha "$candidate_dir")
baseline_compiler_sha=$(sha256sum "$baseline_dir/build/fo/bin/ffc" | \
    awk '{ print $1 }')
candidate_compiler_sha=$(sha256sum "$candidate_dir/build/fo/bin/ffc" | \
    awk '{ print $1 }')

write_report() {
    echo "# ffc 5,000-line lowering benchmark"
    echo
    echo "- UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "- Host: $(uname -srmo)"
    echo "- CPU: $cpu_model"
    echo "- gfortran: $gfortran_version at \`$gfortran_path\` (sha256 \`$gfortran_sha\`)"
    echo "- Baseline: \`$baseline_revision\`, content sha256 \`$baseline_content_sha\`, compiler sha256 \`$baseline_compiler_sha\`"
    echo "- Candidate: \`$candidate_revision\`, content sha256 \`$candidate_content_sha\`, compiler sha256 \`$candidate_compiler_sha\`"
    echo "- FortFront: \`$fortfront_revision\`, index tree \`$fortfront_tree\`"
    echo "- Generated source: $generated_lines lines, $function_count procedures, 3 called sentinels, sha256 \`$source_sha\`"
    echo "- Protocol: one warm-up per compiler, $repeats alternating measured runs, isel backend"
    echo "- Oracle: every generated executable's stdout matched gfortran byte-for-byte"
    echo
    echo "| compiler | median compile wall (s) | median peak RSS (KiB) |"
    echo "|---|---:|---:|"
    echo "| baseline | $baseline_time | $baseline_rss |"
    echo "| candidate | $candidate_time | $candidate_rss |"
    echo
    echo "Candidate compile speedup: ${speedup}x. Peak RSS change: ${rss_change}%."
    echo
    echo "## Raw measurements"
    echo
    echo '```tsv'
    cat "$raw_file"
    echo '```'
}

if [[ -n $report_file ]]; then
    write_report > "$report_file"
    echo "wrote $report_file"
else
    write_report
fi
