# ffc Project Guide

## Mission Statement

**ffc** is a production Fortran compiler that directly translates Fortran AST to HLFIR (High-Level Fortran IR) using MLIR's C API, producing optimized native executables through LLVM's backend.

## Core Principles

### HLFIR-First Compilation Strategy
- **ALWAYS use HLFIR** operations wherever possible
- **FIR only as fallback** when HLFIR lacks specific operation
- **Let LLVM handle lowering** - trust the optimization pipeline
- **Progressive capability** - start simple, add complexity incrementally

## Project Goals

### Immediate (Week 1)
- ✅ Compile empty program to executable
- ✅ Generate readable LLVM IR (.ll files)
- ✅ Establish complete compilation pipeline

### Short Term (Month 1)
- ✅ Compile "Hello, World!" program
- ✅ Support basic variables and arithmetic
- ✅ Handle simple control flow (if/do)
- ✅ Arrays and basic operations

### Medium Term (Month 3)
- ✅ Full procedure support (subroutines/functions)
- ✅ Intrinsic functions
- ✅ Dynamic memory (allocatable arrays)
- ✅ Derived types

### Long Term (Month 6)
- ✅ Module system
- ✅ Lazy Fortran (.lf) support
- ✅ Optimization passes (-O1/-O2)
- ✅ Production-ready compiler

## Architecture Overview

```
Source Files (.f90/.lf) → [fortfront AST] → [HLFIR Builder] → [MLIR Module] → [LLVM] → Executable
```

**Current Status**: Infrastructure complete (51 modules, 82 tests), blocked on fortfront AST access

## Project Structure

**ffc** is a Fortran compiler with MLIR backend using HLFIR (High-Level Fortran IR).
- **51 Fortran modules** + **1 C file** = Complete infrastructure
- **82 active tests** (infrastructure) + **24 disabled tests** (real functionality)
- **Architecture**: Fortran AST → HLFIR → FIR → LLVM IR → Object code

## Core Modules by Category

### 🔧 MLIR C API Foundation (`src/mlir_c/`)
- **`mlir_c_core.f90`** - MLIR context, module, location management
- **`mlir_c_types.f90`** - Type system (integer, float, array, reference types)
- **`mlir_c_attributes.f90`** - Attribute system (integer, float, string attributes)
- **`mlir_c_operations.f90`** - Operation builder infrastructure
- **`mlir_c_stubs.c`** - 🔴 **CRITICAL**: C stub implementations (ALL DUMMY)

### 🏗️ Dialect Bindings (`src/dialects/`)
- **`fir_dialect.f90`** - FIR operations (fir.declare, fir.load, fir.store, etc.)
- **`hlfir_dialect.f90`** - HLFIR operations (hlfir.declare, hlfir.assign, etc.)
- **`standard_dialects.f90`** - Standard dialects (func, arith, scf)

### 🎯 IR Builder Infrastructure (`src/builder/`)
- **`mlir_builder.f90`** - Builder context, insertion points, scopes
- **`ssa_manager.f90`** - SSA value tracking and naming
- **`fortfc_type_converter.f90`** - Fortran to MLIR type conversion
- **`type_conversion_helpers.f90`** - Type helper functions

### 📝 Code Generation (`src/codegen/`)
- **`program_gen.f90`** - Program and module structure generation
- **`function_gen.f90`** - Function and subroutine generation
- **`statement_gen.f90`** - Statement generation (if, do, select, etc.)
- **`expression_gen.f90`** - Expression evaluation and generation

### 🔄 Pass Management (`src/passes/`)
- **`pass_manager.f90`** - MLIR pass manager integration
- **`lowering_pipeline.f90`** - HLFIR→FIR→LLVM lowering pipeline

### 💾 Backend Integration (`src/backend/`)
- **`mlir_c_backend.f90`** - Main compilation backend using C API
- **`compilation_context.f90`** - Compilation state management

### 🛠️ Utilities (`src/utils/`)
- **`memory_tracker.f90`** - Memory leak detection and tracking
- **`memory_guard.f90`** - RAII-style resource management
- **`resource_manager.f90`** - Resource cleanup coordination
- **`performance_tracker.f90`** - Performance measurement and profiling
- **`error_handling.f90`** - Error reporting and diagnostics

### 📊 Test Infrastructure (`test/`)
- **`comprehensive_test_runner.f90`** - Main test harness
- **`run_all_tests.f90`** - Test execution coordinator
- **`test_harness.f90`** - Test framework utilities
- **`performance_benchmarks.f90`** - Performance benchmarking

### 🖥️ Command Line Interface (`app/`)
- **`ffc.f90`** - Main CLI application entry point

## Test Organization

### ✅ Active Tests (82) - Infrastructure Validation
- `test/mlir/test_mlir_context.f90` - MLIR context management
- `test/types/test_type_conversion.f90` - Type system validation
- `test/builder/test_mlir_builder.f90` - IR builder functionality
- `test/codegen/test_program_generation.f90` - Code generation patterns
- `test/backend/test_compilation_pipeline.f90` - Backend integration
- `test/utils/test_memory_management.f90` - Memory tracking

### ❌ Disabled Tests (24) - Real Compiler Functionality
- `test/mlir/test_hlfir_generation.f90.disabled` - 🔴 Real HLFIR operations
- `test/mlir/test_llvm_to_object.f90.disabled` - 🔴 Object file generation
- `test/mlir/test_object_to_executable.f90.disabled` - 🔴 Executable linking
- `test/integration/test_end_to_end_compilation.f90.disabled` - 🔴 Full pipeline
- `test/fortran/test_hello_world.f90.disabled` - 🔴 Hello World compilation

## Build System

### Configuration Files
- **`CMakeLists.txt`** - Main CMake configuration (C++/Fortran mixed)
- **`fpm.toml`** - Fortran Package Manager configuration
- **`configure_build.sh`** - Build setup script with dependency checking

### Dependencies
- **stdlib** ✅ - Available in `build/dependencies/stdlib/`
- **json-fortran** ✅ - Available in `build/dependencies/json-fortran/`
- **fortfront** ✅ - Available in `../fortfront/` (AST parser)

## Documentation (`docs/`)
- **`C_API_USAGE.md`** - MLIR C API usage patterns
- **`DEVELOPER_GUIDE.md`** - Development workflow and architecture
- **`API_REFERENCE.md`** - Complete API documentation
- **`MIGRATION_GUIDE.md`** - Text-to-C-API migration guide
- **`TYPE_CONVERSION.md`** - Type conversion specifications

## Current Status Summary

### ✅ What Works (Infrastructure)
- Complete MLIR C API Fortran wrapper infrastructure
- Full type conversion system (Fortran → MLIR types)
- Builder patterns and SSA value management
- Pass manager framework
- Memory management and resource cleanup
- Comprehensive test harness
- Build system with LLVM/MLIR detection

### 🔴 What's Missing (Core Functionality)
- **`src/mlir_c/mlir_c_stubs.c`**: All C functions return dummy pointers
- Real HLFIR operation generation
- Actual LLVM code generation
- AST parsing integration (fortfront available but not integrated)
- Object file and executable generation

### 🎯 Next Critical Step
Replace `mlir_c_stubs.c` with real MLIR C API implementations to enable actual Fortran compilation.

## Development Workflow

- Run ffc with fpm run. Run tests wirh fpm test