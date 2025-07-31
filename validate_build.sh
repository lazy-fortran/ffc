#!/bin/bash

# ffc Build Validation Script
# Validates build configuration and runs basic tests

set -e

echo "=== ffc Build Validation ==="
echo "Timestamp: $(date)"
echo

# Function to run command and report result
run_test() {
    local test_name="$1"
    local command="$2"
    echo -n "Testing $test_name... "
    
    if eval "$command" > /dev/null 2>&1; then
        echo "✅ PASS"
        return 0
    else
        echo "❌ FAIL"
        return 1
    fi
}

# Function to run command and capture output
run_with_output() {
    local test_name="$1"
    local command="$2"
    echo "Testing $test_name..."
    
    if eval "$command"; then
        echo "✅ $test_name completed successfully"
        return 0
    else
        echo "❌ $test_name failed"
        return 1
    fi
}

# Change to ffc directory
cd "$(dirname "$0")"
echo "Working directory: $(pwd)"
echo

# Test 1: Check basic build
echo "=== Build Tests ==="
run_test "Basic build" "fmp build"

# Test 2: Check build profiles
run_test "Debug profile build" "fpm build --profile debug"
run_test "Release profile build" "fpm build --profile release"

echo

# Test 3: Run core tests
echo "=== Core Test Suite ==="
run_test "MLIR C API core tests" "fpm test test_mlir_c_core"
run_test "Type factory tests" "fpm test test_mlir_c_type_factory" 
run_test "Backend factory tests" "fpm test test_backend_factory"

echo

# Test 4: Check documentation exists
echo "=== Documentation Tests ==="
run_test "BACKLOG.md exists" "test -f BACKLOG.md"
run_test "TEST_STRUCTURE.md exists" "test -f TEST_STRUCTURE.md"
run_test "BUILD_OPTIMIZATION.md exists" "test -f BUILD_OPTIMIZATION.md"
run_test "EPIC4_ANALYSIS.md exists" "test -f EPIC4_ANALYSIS.md"

echo

# Test 5: Source code validation
echo "=== Source Code Validation ==="
run_test "Logger module exists" "test -f src/utils/logger.f90"
run_test "Error handling module exists" "test -f src/utils/error_handling.f90"
run_test "No print statements in core modules" "! grep -r 'print \*' src/mlir_c/ src/builder/ src/dialects/"

echo

# Test 6: Build profile validation  
echo "=== Build Profile Validation ==="
if grep -q "\[profiles\.release\]" fpm.toml; then
    echo "✅ Release profile configured"
else
    echo "❌ Release profile missing"
fi

if grep -q "\[profiles\.debug\]" fpm.toml; then
    echo "✅ Debug profile configured" 
else
    echo "❌ Debug profile missing"
fi

if grep -q "\[profiles\.test\]" fpm.toml; then
    echo "✅ Test profile configured"
else
    echo "❌ Test profile missing"
fi

echo

# Test 7: Count active vs disabled tests
echo "=== Test Organization Validation ==="
active_tests=$(find test/ -name "*.f90" -not -name "*.disabled" | wc -l)
disabled_tests=$(find test/ -name "*.disabled" | wc -l)
total_tests=$((active_tests + disabled_tests))

echo "Active tests: $active_tests"
echo "Disabled tests: $disabled_tests" 
echo "Total tests: $total_tests"

if [ "$total_tests" -eq 88 ]; then
    echo "✅ Test count matches documentation (88 total)"
else
    echo "❌ Test count mismatch (expected 88, found $total_tests)"
fi

echo

# Summary
echo "=== Validation Summary ==="
echo "Build validation completed at $(date)"
echo "ffc project ready for continued development"
echo "Note: Some tests may fail if fortfront is being updated in parallel"