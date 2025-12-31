# FortFC - Fortran Compiler with MLIR C API

**FortFC** is a modern Fortran compiler that generates HLFIR (High-Level FIR) using the MLIR C API exclusively for optimal performance and memory safety.

## Overview

FortFC is a complete Fortran compilation pipeline that transforms Fortran source code into optimized executables through the following stages:

```
Fortran AST → HLFIR (C API) → FIR (C API) → LLVM IR → Object Code
```

**Key Features:**
- **MLIR C API Exclusive**: All MLIR operations created in-memory using C API (no text generation)
- **HLFIR First**: Generate high-level Fortran IR for better optimization
- **Memory Safe**: RAII patterns with automatic resource management
- **Test-Driven**: Comprehensive TDD methodology with RED-GREEN-REFACTOR cycles
- **Performance Focused**: Optimized C API usage with minimal overhead

## Architecture

### Core Components

- **MLIR C Bindings** (`src/mlir_c/`): Low-level MLIR C API Fortran interfaces
- **Dialect Support** (`src/dialects/`): HLFIR, FIR, and standard dialect operations
- **IR Builder** (`src/builder/`): High-level MLIR construction with type conversion
- **Code Generation** (`src/codegen/`): AST to HLFIR transformation
- **Pass Management** (`src/passes/`): HLFIR→FIR→LLVM lowering pipelines
- **Backend Integration** (`src/backend/`): Complete MLIR C API backend
- **Memory Management** (`src/utils/`): Resource tracking and leak detection

## Building

### Prerequisites

- Fortran compiler (gfortran 9+ or ifort)
- LLVM/MLIR development libraries (14+)
- [fortfront](https://github.com/lazy-fortran/fortfront) frontend
- CMake 3.15+ (for C++ components)

### Build Process

```bash
# Build FortFC with fpm
fpm build

# Run comprehensive tests
fpm test

# Run performance benchmarks
./test/performance_benchmarks
```

## Usage

### Basic Compilation

```bash
# Compile Fortran program to executable
ffc program.f90 -o program

# Compile with optimization
ffc program.f90 -O3 -o program
```

### Code Generation Options

```bash
# Generate HLFIR (High-Level FIR)
ffc program.f90 --emit-hlfir

# Generate FIR (Fortran IR)
ffc program.f90 --emit-fir

# Generate LLVM IR
ffc program.f90 --emit-llvm -o program.ll

# Generate object file
ffc program.f90 -c -o program.o
```

### Debug and Analysis

```bash
# Enable memory tracking
ffc program.f90 --debug-memory

# Verbose compilation output
ffc program.f90 --verbose

# Dump MLIR operations
ffc program.f90 --dump-mlir
```

## Development

### Test-Driven Development

FortFC follows strict TDD methodology:

1. **RED Phase**: Write failing tests first
2. **GREEN Phase**: Implement minimal working solution
3. **REFACTOR Phase**: Improve implementation while keeping tests green

```bash
# Run all tests with detailed output
./test/comprehensive_test_runner

# Run specific test categories
./test/test_memory_management
./test/test_type_conversion_validation
./test/performance_benchmarks
```

### Memory Management

All MLIR resources use RAII patterns:

```fortran
use memory_guard

type(memory_guard_t) :: guard
call guard%init()

context = create_mlir_context()
call guard%register_resource(context, "context")

! Automatic cleanup on scope exit
```

### API Documentation

- **[C API Usage Guide](docs/C_API_USAGE.md)**: Complete MLIR C API usage patterns
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)**: Development workflow and architecture
- **[API Reference](docs/API_REFERENCE.md)**: Complete API documentation
- **[Migration Guide](docs/MIGRATION_GUIDE.md)**: Migrating from text-based generation

## Dependencies

- **[fortfront](https://github.com/lazy-fortran/fortfront)**: Frontend AST analysis
- **LLVM/MLIR 14+**: Core MLIR infrastructure and C API
- **gfortran 9+**: Fortran compiler with modern standards support

## Status

### Completed (✅)
- **MLIR C API Foundation**: Standalone implementation with full type/attribute/operation support
- **Dialect Support**: HLFIR, FIR, and standard dialects (func, arith, scf, memref)
- **IR Builder**: High-level MLIR construction with type conversion
- **Pass Management**: HLFIR→FIR→LLVM lowering pipelines
- **Backend Integration**: Complete backend with memory management

### In Progress (🟡)
- **Full HLFIR Code Generation**: AST→HLFIR transformation pipeline
- **CI/CD Integration**: Automated testing and deployment

## Contributing

1. Follow TDD methodology (RED-GREEN-REFACTOR)
2. Use MLIR C API exclusively (no text generation)
3. Implement RAII memory management patterns
4. Write comprehensive tests with performance benchmarks
5. Document all public APIs

See [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for detailed contribution guidelines.

## License

MIT License - see LICENSE file for details.
