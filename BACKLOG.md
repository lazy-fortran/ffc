# ffc Development Backlog

## Overview

This backlog organizes development into atomic issues that each leave ffc in a working state. Issues are prioritized to achieve incremental compilation capability, starting with the absolute minimum and building systematically.

**HARD RESTRICTION**: Always use HLFIR wherever possible, use FIR only when HLFIR unavailable, let LLVM handle lowering.

## Current Status

### ✅ Infrastructure Complete
- 51 Fortran modules with complete MLIR C API wrappers
- 82 infrastructure tests passing
- Real MLIR integration validated (context, module, types working)
- Build system ready (CMake with LLVM detection)

### 🚧 Blocked on fortfront
- Issue #32: Need node accessor functions to read AST
- Cannot proceed with HLFIR generation until AST access restored

### 🎯 Next Milestone
Compile `program hello; end program` to executable

---

## Phase 1: Minimal Compilation Foundation (PRIORITY: CRITICAL)

### Issue #1: Integrate fortfront AST parser [8 points] 🚧 BLOCKED
**Goal**: Connect fortfront to ffc for AST parsing
**Blockers**: fortfront #32 (node accessor functions)
**Tasks**:
- [ ] Create `src/frontend/fortfront_bridge.f90`
- [ ] Implement AST traversal visitor pattern
- [ ] Add node type identification
- [ ] Create test for parsing empty program
**Verification**: Can parse `program hello; end program` and traverse AST

### Issue #2: Generate empty HLFIR module [5 points]
**Goal**: Create minimal MLIR module structure
**Depends on**: #1
**Tasks**:
- [ ] Implement `mlir_module_from_program()` 
- [ ] Create func.func for main program
- [ ] Add module verification
- [ ] Test HLFIR text output with `--emit-hlfir`
**Verification**: `ffc empty.f90 --emit-hlfir` produces valid HLFIR

### Issue #3: Implement LLVM lowering pipeline [8 points]
**Goal**: Lower HLFIR → FIR → LLVM → executable
**Depends on**: #2
**Tasks**:
- [ ] Configure HLFIR-to-FIR pass
- [ ] Configure FIR-to-LLVM pass
- [ ] Integrate LLVM code generation
- [ ] Link to create executable
**Verification**: `ffc empty.f90 -o empty && ./empty` exits cleanly

### Issue #4: Add LLVM IR text output [3 points]
**Goal**: Enable .ll file generation for debugging
**Depends on**: #3
**Tasks**:
- [ ] Implement `--emit-llvm` flag
- [ ] Add LLVM IR printer integration
- [ ] Create .ll file writer
- [ ] Test readable output
**Verification**: `ffc empty.f90 --emit-llvm -o empty.ll` creates readable LLVM IR

### Issue #5: Clean up legacy code [5 points]
**Goal**: Remove text-based MLIR generation
**Depends on**: #3
**Tasks**:
- [ ] Remove all string-based MLIR generation
- [ ] Delete obsolete text generation modules
- [ ] Update all tests to use C API
- [ ] Verify no text generation remains
**Verification**: No MLIR text generation in codebase

---

## Phase 2: Print Statement Support (Hello World!)

### Issue #6: Add string literal support [5 points]
**Goal**: Handle character constants in AST
**Depends on**: #1
**Tasks**:
- [ ] Map character literals to MLIR strings
- [ ] Create global string constants
- [ ] Implement string type conversion
- [ ] Test string literal handling
**Verification**: String literals correctly represented in HLFIR

### Issue #7: Integrate Fortran runtime for I/O [8 points]
**Goal**: Link with Fortran runtime library
**Depends on**: #3
**Tasks**:
- [ ] Locate Fortran runtime library
- [ ] Add runtime linking to build
- [ ] Create runtime interface module
- [ ] Test runtime availability
**Verification**: Can link with Fortran runtime

### Issue #8: Generate print statement HLFIR [8 points]
**Goal**: Compile `print *, "Hello"` statements
**Depends on**: #6, #7
**Tasks**:
- [ ] Map print AST node to runtime call
- [ ] Generate _FortranAioOutputAscii call
- [ ] Handle format descriptors
- [ ] Test print output
**Verification**: `program hello; print *, "Hello, World!"; end program` prints correctly

---

## Phase 3: Variable Support

### Issue #9: Implement hlfir.declare for variables [8 points]
**Goal**: Generate variable declarations
**Depends on**: #2
**Tasks**:
- [ ] Create hlfir.declare operation builder
- [ ] Map Fortran types to MLIR types
- [ ] Handle variable attributes
- [ ] Test declaration generation
**Verification**: Variables properly declared in HLFIR

### Issue #10: Support integer variables [5 points]
**Goal**: Handle integer declarations and literals
**Depends on**: #9
**Tasks**:
- [ ] Map INTEGER to i32/i64
- [ ] Handle integer literals
- [ ] Support integer initialization
- [ ] Test integer operations
**Verification**: `integer :: i = 42` compiles correctly

### Issue #11: Support real variables [5 points]
**Goal**: Handle real declarations and literals
**Depends on**: #9
**Tasks**:
- [ ] Map REAL to f32/f64
- [ ] Handle real literals
- [ ] Support real initialization
- [ ] Test real operations
**Verification**: `real :: x = 3.14` compiles correctly

### Issue #12: Support logical variables [3 points]
**Goal**: Handle logical declarations
**Depends on**: #9
**Tasks**:
- [ ] Map LOGICAL to i1
- [ ] Handle .true./.false.
- [ ] Test logical operations
**Verification**: `logical :: flag = .true.` compiles correctly

### Issue #13: Support character variables [5 points]
**Goal**: Handle character declarations
**Depends on**: #9
**Tasks**:
- [ ] Map CHARACTER to !fir.char
- [ ] Handle string length
- [ ] Support character initialization
- [ ] Test character operations
**Verification**: `character(len=10) :: name = "test"` compiles correctly

---

## Phase 4: Basic Operations

### Issue #14: Implement binary arithmetic operations [8 points]
**Goal**: Support +, -, *, /
**Depends on**: #10, #11
**Tasks**:
- [ ] Generate arith dialect operations
- [ ] Handle type coercion
- [ ] Support mixed integer/real
- [ ] Test arithmetic expressions
**Verification**: `x = 2 + 3 * 4` compiles correctly

### Issue #15: Implement hlfir.assign [8 points]
**Goal**: Support assignment statements
**Depends on**: #9
**Tasks**:
- [ ] Create hlfir.assign builder
- [ ] Handle scalar assignments
- [ ] Support type conversion
- [ ] Test assignments
**Verification**: `x = y + 1` compiles correctly

### Issue #16: Support parenthesized expressions [3 points]
**Goal**: Handle expression precedence
**Depends on**: #14
**Tasks**:
- [ ] Traverse parenthesis nodes
- [ ] Maintain evaluation order
- [ ] Test complex expressions
**Verification**: `x = (a + b) * c` compiles correctly

---

## Phase 5: Control Flow

### Issue #17: Implement if-then statements [8 points]
**Goal**: Support conditional execution
**Depends on**: #12
**Tasks**:
- [ ] Generate scf.if operations
- [ ] Handle condition evaluation
- [ ] Create block structure
- [ ] Test if statements
**Verification**: `if (x > 0) print *, "positive"` compiles

### Issue #18: Support if-then-else [5 points]
**Goal**: Add else branches
**Depends on**: #17
**Tasks**:
- [ ] Add else block generation
- [ ] Handle both branches
- [ ] Test if-else statements
**Verification**: `if (x > 0) then ... else ... end if` compiles

### Issue #19: Implement do loops [8 points]
**Goal**: Support iterative loops
**Depends on**: #10
**Tasks**:
- [ ] Generate scf.for operations
- [ ] Handle loop variables
- [ ] Support loop bounds
- [ ] Test do loops
**Verification**: `do i = 1, 10 ... end do` compiles

---

## Phase 6: Arrays

### Issue #20: Support fixed-size array declarations [8 points]
**Goal**: Handle static arrays
**Depends on**: #9
**Tasks**:
- [ ] Generate !fir.array types
- [ ] Handle array shapes
- [ ] Support array initialization
- [ ] Test array declarations
**Verification**: `integer :: arr(10)` compiles correctly

### Issue #21: Implement hlfir.designate for indexing [8 points]
**Goal**: Support array element access
**Depends on**: #20
**Tasks**:
- [ ] Create hlfir.designate builder
- [ ] Handle array subscripts
- [ ] Support multi-dimensional arrays
- [ ] Test array indexing
**Verification**: `arr(i) = 42` compiles correctly

### Issue #22: Support array sections [8 points]
**Goal**: Handle array slicing
**Depends on**: #21
**Tasks**:
- [ ] Generate bounds for designate
- [ ] Handle stride specification
- [ ] Support partial sections
- [ ] Test array slicing
**Verification**: `arr(1:5) = 0` compiles correctly

### Issue #23: Implement hlfir.elemental [8 points]
**Goal**: Support array operations
**Depends on**: #20
**Tasks**:
- [ ] Create hlfir.elemental builder
- [ ] Generate index functions
- [ ] Handle array expressions
- [ ] Test elemental operations
**Verification**: `arr = arr + 1` compiles correctly

---

## Phase 7: Procedures

### Issue #24: Support subroutine definitions [8 points]
**Goal**: Handle subroutine compilation
**Depends on**: #2
**Tasks**:
- [ ] Generate func.func for subroutines
- [ ] Handle parameter lists
- [ ] Support local variables
- [ ] Test subroutine compilation
**Verification**: `subroutine mysub() ... end subroutine` compiles

### Issue #25: Implement subroutine calls [5 points]
**Goal**: Generate call statements
**Depends on**: #24
**Tasks**:
- [ ] Generate func.call operations
- [ ] Handle argument passing
- [ ] Support by-reference semantics
- [ ] Test subroutine calls
**Verification**: `call mysub()` compiles correctly

### Issue #26: Support function definitions [8 points]
**Goal**: Handle function compilation
**Depends on**: #24
**Tasks**:
- [ ] Handle result variables
- [ ] Generate return operations
- [ ] Support function types
- [ ] Test function compilation
**Verification**: `function myfunc() ... end function` compiles

### Issue #27: Implement function calls [5 points]
**Goal**: Handle function invocation
**Depends on**: #26
**Tasks**:
- [ ] Generate function calls
- [ ] Handle return values
- [ ] Support in expressions
- [ ] Test function calls
**Verification**: `x = myfunc()` compiles correctly

---

## Phase 8: Intrinsic Functions

### Issue #28: Support basic math intrinsics [8 points]
**Goal**: Handle sin, cos, sqrt, etc.
**Depends on**: #27
**Tasks**:
- [ ] Identify intrinsic functions
- [ ] Map to runtime calls
- [ ] Handle type-specific versions
- [ ] Test intrinsic calls
**Verification**: `x = sin(y)` compiles correctly

### Issue #29: Support min/max intrinsics [5 points]
**Goal**: Handle min/max functions
**Depends on**: #27
**Tasks**:
- [ ] Generate comparison operations
- [ ] Handle multiple arguments
- [ ] Support type variants
- [ ] Test min/max
**Verification**: `x = max(a, b, c)` compiles correctly

### Issue #30: Support array intrinsics [8 points]
**Goal**: Handle sum, product, etc.
**Depends on**: #23
**Tasks**:
- [ ] Generate reduction operations
- [ ] Handle array arguments
- [ ] Support dimensional reduction
- [ ] Test array intrinsics
**Verification**: `total = sum(arr)` compiles correctly

---

## Phase 9: Advanced Arrays

### Issue #31: Support allocatable arrays [13 points]
**Goal**: Handle dynamic arrays
**Depends on**: #20
**Tasks**:
- [ ] Generate !fir.box types
- [ ] Implement allocation
- [ ] Handle deallocation
- [ ] Test allocatable arrays
**Verification**: `allocatable :: arr(:)` works correctly

### Issue #32: Implement array allocation [8 points]
**Goal**: Handle allocate statements
**Depends on**: #31
**Tasks**:
- [ ] Generate fir.allocmem
- [ ] Update descriptors
- [ ] Handle allocation errors
- [ ] Test allocation
**Verification**: `allocate(arr(n))` compiles correctly

### Issue #33: Support assumed-shape arrays [8 points]
**Goal**: Handle dummy array arguments
**Depends on**: #31
**Tasks**:
- [ ] Generate descriptor passing
- [ ] Extract bounds from descriptors
- [ ] Handle in procedures
- [ ] Test assumed-shape
**Verification**: Procedures with `(:)` arrays work

---

## Phase 10: Derived Types

### Issue #34: Support type definitions [8 points]
**Goal**: Handle user-defined types
**Depends on**: #9
**Tasks**:
- [ ] Generate !fir.type
- [ ] Map components
- [ ] Handle type names
- [ ] Test type definitions
**Verification**: `type :: point; real :: x, y; end type` compiles

### Issue #35: Support type variables [5 points]
**Goal**: Handle derived type variables
**Depends on**: #34
**Tasks**:
- [ ] Declare type variables
- [ ] Handle initialization
- [ ] Support in procedures
- [ ] Test type variables
**Verification**: `type(point) :: p` compiles correctly

### Issue #36: Implement component access [5 points]
**Goal**: Handle %component syntax
**Depends on**: #35
**Tasks**:
- [ ] Generate hlfir.designate for components
- [ ] Handle nested access
- [ ] Support assignment
- [ ] Test component access
**Verification**: `p%x = 1.0` compiles correctly

---

## Phase 11: Modules

### Issue #37: Support module compilation [13 points]
**Goal**: Compile module units
**Depends on**: #2
**Tasks**:
- [ ] Generate module structure
- [ ] Export symbols
- [ ] Create .mod files
- [ ] Test module compilation
**Verification**: `module mymod ... end module` compiles

### Issue #38: Implement use statements [8 points]
**Goal**: Import module symbols
**Depends on**: #37
**Tasks**:
- [ ] Read .mod files
- [ ] Import symbols
- [ ] Handle renaming
- [ ] Test use statements
**Verification**: `use mymod` works correctly

---

## Phase 12: Lazy Fortran Support

### Issue #39: Add .lf file support [3 points]
**Goal**: Recognize lazy Fortran files
**Depends on**: #1
**Tasks**:
- [ ] Handle .lf extension
- [ ] Enable type inference mode
- [ ] Test .lf parsing
**Verification**: ffc accepts .lf files

### Issue #40: Implement implicit variable declaration [8 points]
**Goal**: Infer types from usage
**Depends on**: #39
**Tasks**:
- [ ] Analyze variable usage
- [ ] Infer types
- [ ] Insert hlfir.declare
- [ ] Test inference
**Verification**: `x = 2` infers integer type

### Issue #41: Support implicit program wrapper [5 points]
**Goal**: Add program structure automatically
**Depends on**: #39
**Tasks**:
- [ ] Detect missing program statement
- [ ] Wrap in program/end program
- [ ] Handle implicit none
- [ ] Test wrapper generation
**Verification**: Bare statements compile in .lf files

### Issue #42: Implement smart type selection [8 points]
**Goal**: Choose optimal types from context
**Depends on**: #40
**Tasks**:
- [ ] Analyze all usage contexts
- [ ] Select appropriate types
- [ ] Handle ambiguous cases
- [ ] Test type selection
**Verification**: Mixed arithmetic infers correctly

---

## Phase 13: Optimization and Polish

### Issue #43: Add optimization passes [8 points]
**Goal**: Improve generated code quality
**Depends on**: #3
**Tasks**:
- [ ] Add -O optimization flags
- [ ] Configure optimization passes
- [ ] Test performance
**Verification**: -O2 produces faster code

### Issue #44: Improve error messages [5 points]
**Goal**: Better user diagnostics
**Depends on**: #1
**Tasks**:
- [ ] Add source locations
- [ ] Improve error text
- [ ] Add suggestions
- [ ] Test error reporting
**Verification**: Clear error messages with line numbers

### Issue #45: Add compiler driver features [5 points]
**Goal**: Match gfortran CLI interface
**Tasks**:
- [ ] Support common flags
- [ ] Add help text
- [ ] Handle multiple files
- [ ] Test CLI compatibility
**Verification**: Drop-in replacement for simple cases

---

## Testing Requirements

Each issue must include:
1. Unit tests for new functionality
2. Integration test with complete program
3. Comparison with gfortran/flang output (where applicable)
4. Documentation update
5. Example program demonstrating capability

## Success Metrics

### Immediate (Phase 1-2)
- ✅ Compiles empty program
- ✅ Compiles Hello World
- ✅ Generates working executables
- ✅ Produces readable .ll files

### Short Term (Phase 3-5)
- ✅ Basic arithmetic works
- ✅ Variables and assignments work
- ✅ Simple control flow works
- ✅ All tests pass

### Medium Term (Phase 6-9)
- ✅ Arrays work correctly
- ✅ Procedures compile
- ✅ Intrinsics supported
- ✅ Performance acceptable

### Long Term (Phase 10-13)
- ✅ Derived types work
- ✅ Modules compile
- ✅ Lazy Fortran works
- ✅ Production ready

---

## Notes

- **Priority**: Issues numbered in dependency order
- **Points**: Rough complexity estimate (Fibonacci scale)
- **Atomic**: Each issue leaves system working
- **Testable**: Each issue has clear verification
- **HLFIR-first**: Always prefer HLFIR operations