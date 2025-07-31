cmake_minimum_required(VERSION 3.15)
project(ffc VERSION 0.1.0 LANGUAGES Fortran C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Try to find LLVM/MLIR - graceful fallback if not found
find_package(LLVM CONFIG QUIET)
find_package(MLIR CONFIG QUIET)

if(LLVM_FOUND)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
    set(HAVE_LLVM TRUE)
else()
    message(STATUS "LLVM not found - using stub implementation")
    set(HAVE_LLVM FALSE)
endif()

if(MLIR_FOUND)
    message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
    set(HAVE_MLIR TRUE)
else()
    message(STATUS "MLIR not found - using stub implementation")
    set(HAVE_MLIR FALSE)
endif()

# Include directories (only if found)
if(HAVE_LLVM)
    include_directories(${LLVM_INCLUDE_DIRS})
    add_definitions(${LLVM_DEFINITIONS})
endif()

if(HAVE_MLIR)
    include_directories(${MLIR_INCLUDE_DIRS})
endif()

# Enable mixed language linking
enable_language(Fortran)
enable_language(C)
enable_language(CXX)

# Set Fortran compiler flags
set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0 -fcheck=all -Wall -Wextra")
set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -march=native")

# Create MLIR C API wrapper library
add_library(mlir_c_api SHARED
    src/mlir_c/mlir_c_stubs.c
)

# Link MLIR libraries (only if available)
if(HAVE_MLIR AND HAVE_LLVM)
    target_link_libraries(mlir_c_api
        MLIRC
        MLIRSupport
        MLIRParser
        MLIRIR
        MLIRDialect
        MLIRFuncDialect
        MLIRArithDialect
        MLIRSCFDialect
        MLIRFIRDialect
        MLIRHLFIRDialect
    )
else()
    # For stub implementation, no external libraries needed
    message(STATUS "Using stub MLIR C API implementation")
endif()

# Fortran modules
add_library(ffc_fortran STATIC
    src/mlir_c/mlir_c_core.f90
    src/mlir_c/mlir_c_types.f90
    src/mlir_c/mlir_c_attributes.f90
    src/mlir_c/mlir_c_operations.f90
    src/dialects/fir_dialect.f90
    src/dialects/hlfir_dialect.f90
    src/dialects/standard_dialects.f90
    src/builder/mlir_builder.f90
    src/builder/ssa_manager.f90
    src/builder/fortfc_type_converter.f90
    src/builder/type_conversion_helpers.f90
    src/codegen/program_gen.f90
    src/codegen/function_gen.f90
    src/codegen/statement_gen.f90
    src/codegen/expression_gen.f90
    src/passes/pass_manager.f90
    src/passes/lowering_pipeline.f90
    src/backend/mlir_c_backend.f90
    src/utils/memory_tracker.f90
    src/utils/memory_guard.f90
    src/utils/resource_manager.f90
)

# Link Fortran library with C API
target_link_libraries(ffc_fortran mlir_c_api)

# Main executable
add_executable(ffc
    app/ffc.f90
)

target_link_libraries(ffc
    ffc_fortran
    mlir_c_api
)

# Install targets
install(TARGETS ffc mlir_c_api ffc_fortran
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Test configuration
enable_testing()

# Helper function to add Fortran tests
function(add_fortran_test test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} ffc_fortran mlir_c_api)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Add comprehensive tests
add_fortran_test(test_comprehensive test/comprehensive_test_runner.f90)
add_fortran_test(test_memory_management test/test_memory_management.f90)
add_fortran_test(test_performance test/performance_benchmarks.f90)
add_fortran_test(test_integration test/test_integration_hello_world.f90)

# CTest configuration
set(CTEST_OUTPUT_ON_FAILURE ON)