# fortc

Full compilation backend for lazy fortran - MLIR, HLFIR, LLVM IR, object code generation.

## Overview

fortc is the compilation backend for the lazy-fortran ecosystem that provides:
- MLIR code generation with HLFIR targeting
- LLVM IR emission and optimization
- Object code and executable generation
- Integration with Enzyme for automatic differentiation

## Features

- HLFIR (High-Level Fortran IR) code generation
- FIR (Fortran IR) lowering
- LLVM IR emission and optimization
- Object file and executable generation
- Automatic differentiation support via Enzyme
- Modular backend architecture

## Building

```bash
fpm build
```

## Usage

Compile a Fortran program:
```bash
fortc program.f90 -o program
```

Emit HLFIR code:
```bash
fortc program.f90 --emit-hlfir
```

Emit LLVM IR:
```bash  
fortc program.f90 --emit-llvm -o program.ll
```

Generate object file:
```bash
fortc program.f90 -o program.o
```

## Dependencies

- [fortfront](https://github.com/lazy-fortran/fortfront) - Frontend analysis
- LLVM/MLIR libraries
- Fortran compiler with MLIR support

## Architecture

fortc uses a modular backend architecture:
- `backend_interface` - Abstract backend interface
- `mlir_backend` - MLIR/HLFIR code generation
- `fortran_backend` - Standard Fortran emission
- `backend_factory` - Backend selection and creation

## License

MIT License - see LICENSE file for details.