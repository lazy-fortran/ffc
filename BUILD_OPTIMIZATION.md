# ffc Build Optimization Guide

## Build Profile Improvements Made

Added optimized build profiles to `fpm.toml`:

### Release Profile
```toml
[profiles.release]
flags = "-O3 -march=native"
```
- Maximum optimization level (-O3)
- Native CPU architecture optimization
- Use for production builds

### Debug Profile  
```toml
[profiles.debug]
flags = "-g -O0 -fcheck=all -Wall -Wextra"
```
- Full debug information (-g)
- No optimization (-O0) for debugging clarity
- All runtime checks enabled
- Comprehensive warnings

### Test Profile
```toml
[profiles.test]
flags = "-g -O1 -fcheck=bounds"
```
- Debug info with light optimization
- Bounds checking for safety
- Balanced performance/debugging

## Usage

Build with specific profile:
```bash
fpm build --profile release
fpm test --profile test
fpm build --profile debug
```

## Current Limitations and TODOs

### Epic 4 Implementation Gaps
Found 11 TODO items in backend code:

#### Core Compilation (mlir_compile.f90):
- MLIR compilation pipeline implementation
- MLIR lowering passes implementation

#### I/O Operations (mlir_io_operations.f90):
- Type-based output function selection

#### Backend Generation (mlir_backend.f90):
- Module namespacing (2 items)
- Rename list parsing
- Expression evaluation
- Type-based function resolution (2 items)
- Argument passing for function calls
- Operator resolution
- WHERE construct implementation
- Derived type definition

## Build Performance Characteristics

### Current Build Profile:
- Dependencies: stdlib, fortfront, test-drive, json-fortran
- 88 test files (64 active, 24 disabled)
- C API bindings included (mlir_c_stubs.c)

### Optimization Opportunities:
1. **Profile Usage**: Use release profile for benchmarking
2. **Test Selection**: Use test profile for validation
3. **Debug Analysis**: Use debug profile for troubleshooting
4. **Parallel Building**: FPM supports parallel compilation
5. **Dependency Caching**: Build cache already configured

## Memory and Performance Considerations

### C API Integration:
- mlir_c_stubs.c provides C interface
- Smart pointer management in mlir_c_smart_ptr.f90
- Type factory caching in mlir_c_type_factory.f90

### Architecture:
- Well-organized source structure (mlir_c/, dialects/, builder/, backend/)
- Clean separation of concerns
- Modular design supports incremental compilation

## Next Steps for Full Optimization

1. **Complete Epic 4**: Implement remaining backend functionality
2. **Enable Disabled Tests**: Address underlying issues in 24 disabled tests
3. **Performance Profiling**: Benchmark with release profile
4. **Memory Analysis**: Use debug profile for leak detection
5. **CI Integration**: Configure profiles for different CI stages

## Testing Strategy

Use appropriate profiles for different testing scenarios:
- Development: `fpm test --profile debug`
- CI validation: `fpm test --profile test` 
- Performance testing: `fpm test --profile release`