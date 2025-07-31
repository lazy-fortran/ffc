#!/bin/bash

# FortFC Build Configuration Script
# This script configures the build environment with proper error handling

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for required tools
check_requirements() {
    print_status "Checking build requirements..."
    
    # Check for CMake
    if command -v cmake &> /dev/null; then
        CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
        print_status "Found CMake $CMAKE_VERSION"
    else
        print_error "CMake not found. Please install CMake 3.15+"
        exit 1
    fi
    
    # Check for Fortran compiler
    if command -v gfortran &> /dev/null; then
        GFORTRAN_VERSION=$(gfortran --version | head -n1)
        print_status "Found $GFORTRAN_VERSION"
    elif command -v ifort &> /dev/null; then
        IFORT_VERSION=$(ifort --version | head -n1)
        print_status "Found $IFORT_VERSION"
    else
        print_error "No Fortran compiler found. Please install gfortran or ifort"
        exit 1
    fi
    
    # Check for C++ compiler
    if command -v g++ &> /dev/null; then
        GCC_VERSION=$(g++ --version | head -n1)
        print_status "Found $GCC_VERSION"
    elif command -v clang++ &> /dev/null; then
        CLANG_VERSION=$(clang++ --version | head -n1)
        print_status "Found $CLANG_VERSION"
    else
        print_error "No C++ compiler found. Please install g++ or clang++"
        exit 1
    fi
    
    # Check for LLVM/MLIR (optional)
    if command -v llvm-config &> /dev/null; then
        LLVM_VERSION=$(llvm-config --version)
        print_status "Found LLVM $LLVM_VERSION"
    else
        print_warning "LLVM not found - will use stub implementation"
    fi
}

# Configure build directory
configure_build() {
    local BUILD_TYPE=${1:-Release}
    local BUILD_DIR="build"
    
    print_status "Configuring build (type: $BUILD_TYPE)..."
    
    # Create build directory
    if [ -d "$BUILD_DIR" ]; then
        print_warning "Build directory exists, cleaning..."
        rm -rf "$BUILD_DIR"
    fi
    
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Configure with CMake
    cmake .. \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="../install" \
        -DCMAKE_VERBOSE_MAKEFILE=OFF
    
    print_status "Build configured successfully"
}

# Build the project
build_project() {
    local JOBS=${1:-$(nproc)}
    
    print_status "Building project (jobs: $JOBS)..."
    
    if [ ! -d "build" ]; then
        print_error "Build not configured. Run configure first."
        exit 1
    fi
    
    cd build
    make -j"$JOBS"
    
    print_status "Build completed successfully"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    if [ ! -d "build" ]; then
        print_error "Build not configured. Run configure and build first."
        exit 1
    fi
    
    cd build
    ctest --output-on-failure
    
    print_status "All tests passed"
}

# Main script logic
main() {
    case "${1:-help}" in
        "check")
            check_requirements
            ;;
        "configure")
            check_requirements
            configure_build "${2:-Release}"
            ;;
        "build")
            build_project "${2:-$(nproc)}"
            ;;
        "test")
            run_tests
            ;;
        "all")
            check_requirements
            configure_build "${2:-Release}"
            build_project "$(nproc)"
            run_tests
            ;;
        "clean")
            print_status "Cleaning build directory..."
            rm -rf build install
            print_status "Clean completed"
            ;;
        "help"|*)
            echo "FortFC Build Configuration Script"
            echo ""
            echo "Usage: $0 [command] [options]"
            echo ""
            echo "Commands:"
            echo "  check          Check build requirements"
            echo "  configure      Configure build (optionally specify Debug/Release)"
            echo "  build          Build project (optionally specify job count)"
            echo "  test           Run test suite"
            echo "  all            Check, configure, build, and test"
            echo "  clean          Clean build directory"
            echo "  help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 all                    # Full build and test"
            echo "  $0 configure Debug        # Configure debug build"
            echo "  $0 build 4               # Build with 4 parallel jobs"
            ;;
    esac
}

main "$@"