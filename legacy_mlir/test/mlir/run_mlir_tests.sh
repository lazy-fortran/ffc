#!/bin/bash
# MLIR Test Harness Shell Script
# Provides convenient interface for running MLIR tests

set -e

# Default values
VERBOSE=false
VALIDATE_MLIR=false
FILTER=""
JUNIT_OUTPUT=""
BUILD_FIRST=true

# Function to generate JUnit XML
generate_junit_xml() {
    local output_file="$1"

    cat > "$output_file" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="MLIR Tests" tests="$TOTAL_TESTS" failures="$FAILED_TESTS" time="$TOTAL_TIME">
  <testsuite name="mlir" tests="$TOTAL_TESTS" failures="$FAILED_TESTS" time="$TOTAL_TIME">
EOF

    for result in "${TEST_RESULTS[@]}"; do
        IFS=':' read -r name status time <<< "$result"

        cat >> "$output_file" << EOF
    <testcase name="$name" classname="mlir.$name" time="$time">
EOF

        if [[ "$status" =~ FAIL ]]; then
            cat >> "$output_file" << EOF
      <failure message="Test failed" type="TestFailure">$status</failure>
EOF
        fi

        cat >> "$output_file" << EOF
    </testcase>
EOF
    done

    cat >> "$output_file" << EOF
  </testsuite>
</testsuites>
EOF
}

usage() {
    cat << EOF
Usage: $0 [options] [filter]

Options:
    -v, --verbose       Verbose output
    --validate-mlir     Validate MLIR output with mlir-opt
    --filter PATTERN    Run only tests matching pattern
    --junit FILE        Generate JUnit XML output
    --no-build          Skip building tests
    -h, --help          Show this help

Examples:
    $0                          # Run all tests
    $0 --verbose               # Run with verbose output
    $0 --validate-mlir         # Run with MLIR validation
    $0 --filter enzyme         # Run only enzyme tests
    $0 --junit results.xml     # Generate JUnit XML

Environment Variables:
    OMP_NUM_THREADS            Number of parallel build threads (default: 4)
    MLIR_OPT_PATH             Path to mlir-opt tool (default: mlir-opt)
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --validate-mlir)
            VALIDATE_MLIR=true
            shift
            ;;
        --filter)
            FILTER="$2"
            shift 2
            ;;
        --junit)
            JUNIT_OUTPUT="$2"
            shift 2
            ;;
        --no-build)
            BUILD_FIRST=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            # Treat as filter
            FILTER="$1"
            shift
            ;;
    esac
done

# Set defaults
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
MLIR_OPT_PATH=${MLIR_OPT_PATH:-mlir-opt}

# Check for required tools
if [ "$VALIDATE_MLIR" = true ]; then
    if ! command -v "$MLIR_OPT_PATH" &> /dev/null; then
        echo "Error: mlir-opt not found. Please install MLIR tools or set MLIR_OPT_PATH."
        exit 1
    fi
fi

# Change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== MLIR Test Harness ==="
echo "Project root: $PROJECT_ROOT"
echo "Threads: $OMP_NUM_THREADS"
echo "Verbose: $VERBOSE"
echo "MLIR validation: $VALIDATE_MLIR"
echo "Filter: ${FILTER:-'(none)'}"
echo ""

# Build tests if requested
if [ "$BUILD_FIRST" = true ]; then
    echo "Building tests..."
    if [ "$VERBOSE" = true ]; then
        fpm build
    else
        fpm build > /dev/null 2>&1
    fi
    echo "Build complete."
    echo ""
fi

# Track test results
declare -a TEST_RESULTS=()
declare -a TEST_NAMES=("test_basic_generation" "test_ast_mapping" "test_types" "test_optimization" "test_llvm_lowering" "test_enzyme_ad")
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
START_TIME=$(date +%s.%N)

# Run tests
echo "Running MLIR tests..."
echo ""

for test_name in "${TEST_NAMES[@]}"; do
    # Apply filter if specified
    if [ -n "$FILTER" ] && [[ ! "$test_name" =~ $FILTER ]]; then
        continue
    fi

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    echo -n "Running $test_name... "

    TEST_START=$(date +%s.%N)

    # Run the test
    if [ "$VERBOSE" = true ]; then
        echo ""
        if fpm test "$test_name"; then
            result="PASS"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            result="FAIL"
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
    else
        if fpm test "$test_name" > /dev/null 2>&1; then
            result="PASS"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            result="FAIL"
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
    fi

    TEST_END=$(date +%s.%N)
    TEST_TIME=$(echo "$TEST_END - $TEST_START" | bc -l)

    # MLIR validation if requested
    if [ "$VALIDATE_MLIR" = true ] && [ "$result" = "PASS" ]; then
        # Look for generated MLIR files
        mlir_files=($(find . -name "*.mlir" -newer /tmp/test_start_marker 2>/dev/null || true))

        if [ ${#mlir_files[@]} -gt 0 ]; then
            for mlir_file in "${mlir_files[@]}"; do
                if [ "$VERBOSE" = true ]; then
                    echo "  Validating MLIR: $mlir_file"
                fi

                if ! "$MLIR_OPT_PATH" --verify-each "$mlir_file" > /dev/null 2>&1; then
                    result="FAIL (MLIR validation)"
                    PASSED_TESTS=$((PASSED_TESTS - 1))
                    FAILED_TESTS=$((FAILED_TESTS + 1))
                    break
                fi
            done
        fi
    fi

    if [ "$VERBOSE" = false ]; then
        printf "%-12s (%.3fs)\n" "$result" "$TEST_TIME"
    fi

    # Store result for JUnit output
    TEST_RESULTS+=("$test_name:$result:$TEST_TIME")
done

END_TIME=$(date +%s.%N)
TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc -l)

# Report summary
echo ""
echo "=== Summary ==="
echo "Total tests:  $TOTAL_TESTS"
echo "Passed:       $PASSED_TESTS"
echo "Failed:       $FAILED_TESTS"

if [ $TOTAL_TESTS -gt 0 ]; then
    PASS_RATE=$(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l)
    echo "Pass rate:    ${PASS_RATE}%"
fi

printf "Total time:   %.3fs\n" "$TOTAL_TIME"

echo ""
if [ $FAILED_TESTS -gt 0 ]; then
    echo "Some tests FAILED!"
    exit_code=1
else
    echo "All tests PASSED!"
    exit_code=0
fi

# Generate JUnit XML if requested
if [ -n "$JUNIT_OUTPUT" ]; then
    echo ""
    echo "Generating JUnit XML: $JUNIT_OUTPUT"
    generate_junit_xml "$JUNIT_OUTPUT"
fi

exit $exit_code
