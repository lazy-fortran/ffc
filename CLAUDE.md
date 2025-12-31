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
- Compile empty program to executable
- Generate readable LLVM IR (.ll files)
- Establish complete compilation pipeline

### Short Term (Month 1)
- Compile "Hello, World!" program
- Support basic variables and arithmetic
- Handle simple control flow (if/do)
- Arrays and basic operations

### Medium Term (Month 3)
- Full procedure support (subroutines/functions)
- Intrinsic functions
- Dynamic memory (allocatable arrays)
- Derived types

### Long Term (Month 6)
- Module system
- Lazy Fortran (.lf) support
- Optimization passes (-O1/-O2)
- Production-ready compiler

## Architecture Overview

```
Source Files (.f90/.lf) -> [fortfront AST] -> [HLFIR Builder] -> [MLIR Module] -> [LLVM] -> Executable
```

**Current Status**: MLIR C API infrastructure complete, implementing full HLFIR code generation pipeline

## Project Structure

**ffc** is a Fortran compiler with MLIR backend using HLFIR (High-Level Fortran IR).
- **51+ Fortran modules** + **1 C file** = Complete infrastructure
- **32 passing tests** (infrastructure) + more tests pending HLFIR pipeline
- **Architecture**: Fortran AST -> HLFIR -> FIR -> LLVM IR -> Object code

## Core Modules by Category

### MLIR C API Foundation (`src/mlir_c/`)
- **`mlir_c_core.f90`** - MLIR context, module, location management
- **`mlir_c_types.f90`** - Type system (integer, float, complex, array, reference types)
- **`mlir_c_attributes.f90`** - Attribute system (integer, float, string, array attributes)
- **`mlir_c_operations.f90`** - Operation builder infrastructure
- **`mlir_c_real.c`** - Standalone C implementation of MLIR structures

### Dialect Bindings (`src/dialects/`)
- **`fir_dialect.f90`** - FIR operations (fir.declare, fir.load, fir.store, etc.)
- **`hlfir_dialect.f90`** - HLFIR operations (hlfir.declare, hlfir.assign, etc.)
- **`standard_dialects.f90`** - Standard dialects (func, arith, scf)

### IR Builder Infrastructure (`src/builder/`)
- **`mlir_builder.f90`** - Builder context, insertion points, scopes
- **`ssa_manager.f90`** - SSA value tracking and naming
- **`fortfc_type_converter.f90`** - Fortran to MLIR type conversion
- **`type_conversion_helpers.f90`** - Type helper functions

### Code Generation (`src/codegen/`)
- **`program_gen.f90`** - Program and module structure generation
- **`function_gen.f90`** - Function and subroutine generation
- **`statement_gen.f90`** - Statement generation (if, do, select, etc.)
- **`expression_gen.f90`** - Expression evaluation and generation

### Pass Management (`src/passes/`)
- **`pass_manager.f90`** - MLIR pass manager integration
- **`lowering_pipeline.f90`** - HLFIR->FIR->LLVM lowering pipeline

### Backend Integration (`src/backend/`)
- **`mlir_c_backend.f90`** - Main compilation backend using C API
- **`mlir_backend.f90`** - MLIR backend with text IR generation

### Utilities (`src/utils/`)
- **`memory_tracker.f90`** - Memory leak detection and tracking
- **`memory_guard.f90`** - RAII-style resource management
- **`resource_manager.f90`** - Resource cleanup coordination
- **`performance_tracker.f90`** - Performance measurement and profiling

### Command Line Interface (`app/`)
- **`ffc.f90`** - Main CLI application entry point

## Build System

### Configuration Files
- **`fpm.toml`** - Fortran Package Manager configuration
- **`CMakeLists.txt`** - CMake configuration (optional, for LLVM linking)

### Dependencies
- **stdlib** - Available via fpm
- **fortfront** - Available in `../fortfront/` (AST parser)

## Current Status Summary

### What Works (Infrastructure)
- Complete standalone MLIR C API implementation (`mlir_c_real.c`)
- Full type system (integer, float, complex, memref, tensor, function types)
- Full attribute system (integer, float, string, array, type attributes)
- Operation creation and destruction
- Pass manager framework
- Memory management and resource cleanup
- Dialect registration (HLFIR, FIR, func, arith, scf, memref)
- 32 infrastructure tests passing

### In Progress
- Full HLFIR code generation from fortfront AST
- Integration with fortfront for complete pipeline

### Known Issues
- 6 tests have fortfront destructor segfaults (fortfront issue, not ffc)

## Development Workflow

- Build: `fpm build`
- Test: `fpm test`
- Run: `fpm run`
