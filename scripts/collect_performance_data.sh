#!/bin/bash

# Performance Data Collection Script
# Collects performance data from test runs and benchmarks

set -euo pipefail

# Configuration
PERF_DIR="performance_data"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="$PERF_DIR/performance_report_$TIMESTAMP.json"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create performance data directory
setup_directories() {
    print_status "Setting up performance data collection..."
    mkdir -p "$PERF_DIR"
    mkdir -p "$PERF_DIR/benchmarks"
    mkdir -p "$PERF_DIR/logs"
}

# Run performance benchmarks
run_benchmarks() {
    print_status "Running performance benchmarks..."
    
    local start_time end_time duration
    
    # Comprehensive test suite benchmark
    start_time=$(date +%s.%N)
    if ./test/comprehensive_test_runner > "$PERF_DIR/logs/comprehensive_test_$TIMESTAMP.log" 2>&1; then
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc -l)
        echo "comprehensive_test_suite,$duration" >> "$PERF_DIR/benchmarks/test_times.csv"
        print_status "Comprehensive test suite: ${duration}s"
    else
        print_warning "Comprehensive test suite failed"
    fi
    
    # Memory management benchmark
    start_time=$(date +%s.%N)
    if ./test/test_memory_management > "$PERF_DIR/logs/memory_test_$TIMESTAMP.log" 2>&1; then
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc -l)
        echo "memory_management_test,$duration" >> "$PERF_DIR/benchmarks/test_times.csv"
        print_status "Memory management test: ${duration}s"
    else
        print_warning "Memory management test failed"
    fi
    
    # Integration test benchmark
    start_time=$(date +%s.%N)
    if ./test/test_integration_hello_world > "$PERF_DIR/logs/integration_test_$TIMESTAMP.log" 2>&1; then
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc -l)
        echo "integration_test,$duration" >> "$PERF_DIR/benchmarks/test_times.csv"
        print_status "Integration test: ${duration}s"
    else
        print_warning "Integration test failed"
    fi
    
    # Performance benchmarks
    start_time=$(date +%s.%N)
    if ./test/performance_benchmarks > "$PERF_DIR/logs/perf_benchmarks_$TIMESTAMP.log" 2>&1; then
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc -l)
        echo "performance_benchmarks,$duration" >> "$PERF_DIR/benchmarks/test_times.csv"
        print_status "Performance benchmarks: ${duration}s"
    else
        print_warning "Performance benchmarks failed"
    fi
}

# Collect system information
collect_system_info() {
    print_status "Collecting system information..."
    
    local sys_info_file="$PERF_DIR/system_info_$TIMESTAMP.txt"
    
    {
        echo "=== System Information ==="
        echo "Date: $(date)"
        echo "Hostname: $(hostname)"
        echo "OS: $(uname -a)"
        echo "CPU: $(grep 'model name' /proc/cpuinfo | head -n1 | cut -d: -f2 | xargs)"
        echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
        echo "Compiler: $(gfortran --version | head -n1)"
        echo "CMake: $(cmake --version | head -n1)"
        echo ""
        echo "=== Git Information ==="
        echo "Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
        echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
        echo "Commit Message: $(git log -1 --pretty=format:'%s' 2>/dev/null || echo 'unknown')"
    } > "$sys_info_file"
}

# Generate performance report
generate_report() {
    print_status "Generating performance report..."
    
    # Create JSON report
    cat > "$REPORT_FILE" << EOF
{
  "timestamp": "$TIMESTAMP",
  "date": "$(date -Iseconds)",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
  "system": {
    "os": "$(uname -s)",
    "arch": "$(uname -m)",
    "hostname": "$(hostname)",
    "compiler": "$(gfortran --version | head -n1 | cut -d' ' -f4)"
  },
  "benchmarks": {
EOF

    # Add benchmark data if available
    if [ -f "$PERF_DIR/benchmarks/test_times.csv" ]; then
        echo '    "test_times": [' >> "$REPORT_FILE"
        first=true
        while IFS=',' read -r test_name duration; do
            if [ "$first" = true ]; then
                first=false
            else
                echo ',' >> "$REPORT_FILE"
            fi
            echo -n "      {\"name\": \"$test_name\", \"duration\": $duration}" >> "$REPORT_FILE"
        done < "$PERF_DIR/benchmarks/test_times.csv"
        echo '' >> "$REPORT_FILE"
        echo '    ]' >> "$REPORT_FILE"
    else
        echo '    "test_times": []' >> "$REPORT_FILE"
    fi

    cat >> "$REPORT_FILE" << EOF
  }
}
EOF

    print_status "Performance report generated: $REPORT_FILE"
}

# Compare with previous results
compare_performance() {
    print_status "Comparing with previous performance data..."
    
    local latest_report previous_report
    latest_report=$(ls -t "$PERF_DIR"/performance_report_*.json 2>/dev/null | head -n1)
    previous_report=$(ls -t "$PERF_DIR"/performance_report_*.json 2>/dev/null | head -n2 | tail -n1)
    
    if [ -n "$previous_report" ] && [ "$latest_report" != "$previous_report" ]; then
        print_status "Previous report found: $(basename "$previous_report")"
        # Simple comparison - could be enhanced with proper JSON parsing
        print_status "Performance comparison available in reports"
    else
        print_warning "No previous performance data for comparison"
    fi
}

# Main function
main() {
    case "${1:-all}" in
        "setup")
            setup_directories
            ;;
        "benchmark")
            run_benchmarks
            ;;
        "collect")
            collect_system_info
            generate_report
            ;;
        "compare")
            compare_performance
            ;;
        "all")
            setup_directories
            run_benchmarks
            collect_system_info
            generate_report
            compare_performance
            ;;
        "help"|*)
            echo "Performance Data Collection Script"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  setup      Setup performance data directories"
            echo "  benchmark  Run performance benchmarks"
            echo "  collect    Collect system info and generate report"
            echo "  compare    Compare with previous performance data"
            echo "  all        Run all steps (default)"
            echo "  help       Show this help message"
            ;;
    esac
}

main "$@"