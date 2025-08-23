# ffc Development Roadmap

**HLFIR-FIRST PRINCIPLE**: Always use HLFIR operations, FIR only as fallback, let LLVM handle lowering

## Current Status: Infrastructure Ready, fortfront Integration Simplified

### ✅ Complete
- MLIR C API wrapper infrastructure (51 modules)
- Type conversion system
- Build system with LLVM/MLIR detection
- Test harness (82 infrastructure tests)
- Real MLIR integration validated
- **fpm automatic static linking to fortfront** (configured in fpm.toml)

### 🚧 Blocked
- **fortfront Issue #32**: Need node accessor functions for AST traversal
- Cannot proceed with compilation until AST access restored
- **NOTE**: fmp handles fortfront dependency automatically - no manual linking needed

### 🎯 Immediate Goal
Compile `program hello; end program` to working executable

---

## Phase 0: Unblock Development (CRITICAL - 1 Week)

### Issue #1: Create fortfront stub bridge [3 points]
**Goal**: Mock fortfront interface for development while waiting for real API
**Architecture**: fmp automatically handles static linking to fortfront dependency
**Tasks**:
- Create `src/frontend/fortfront_mock.f90` with hardcoded AST
- Mock parsing `program hello; end program`
- Define AST node interface we need from fortfront
- Document required fortfront API for Issue #32
- Leverage fpm's automatic dependency management (already configured)
**Verification**: Can traverse mock AST structure with automatic fortfront linking

### Issue #2: Design fortfront integration API [2 points]
**Goal**: Define exact interface needed from fortfront
**Architecture**: Integration simplified by fpm's automatic static linking
**Tasks**:
- Document all AST node types we need to access
- Define traversal pattern requirements
- Create interface specification for fortfront team
- Submit as comment on fortfront Issue #32
- Confirm fpm dependency resolution works correctly
**Verification**: Clear specification ready for fortfront with confirmed build integration

---

## Phase 1: Minimal Compilation (Week 1)

### Issue #3: Generate empty HLFIR module [5 points]
**Goal**: Create minimal MLIR module for empty program
**Depends on**: #1
**Tasks**:
- Implement `generate_empty_program()` using real MLIR C API
- Create func.func for main program
- Add module verification
- Generate text output for debugging
**Verification**: Valid HLFIR module from empty program

### Issue #4: Implement HLFIR-to-LLVM pipeline [8 points]
**Goal**: Complete lowering pipeline
**Depends on**: #3
**Tasks**:
- Configure HLFIR→FIR pass (use MLIR's built-in)
- Configure FIR→LLVM pass (use MLIR's built-in)
- Link with LLVM code generation
- Generate executable
**Verification**: `./ffc empty.f90 -o empty && ./empty` exits cleanly

### Issue #5: Add LLVM IR output [3 points]
**Goal**: Enable debugging via .ll files
**Depends on**: #4
**Tasks**:
- Implement `--emit-llvm` flag
- Integrate LLVM IR printer
- Write .ll file to disk
- Add readable formatting
**Verification**: `./ffc empty.f90 --emit-llvm` produces readable LLVM IR

---

## Phase 2: Hello World (Week 2)

### Issue #6: Support string literals [5 points]
**Goal**: Handle character constants
**Depends on**: #3
**Tasks**:
- Map character literals to MLIR string attributes
- Create global string constants
- Implement !fir.char type conversion
- Test with "Hello, World!"
**Verification**: String literals in HLFIR module

### Issue #7: Link Fortran runtime [8 points]
**Goal**: Enable I/O operations
**Depends on**: #4
**Tasks**:
- Locate flang runtime library
- Add runtime linking to CMake
- Create runtime interface module
- Test runtime symbols available
**Verification**: Can link with _FortranAioOutputAscii

### Issue #8: Generate print statement [8 points]
**Goal**: Compile print statements
**Depends on**: #6, #7
**Tasks**:
- Map print AST node to runtime calls
- Generate I/O begin/output/end sequence
- Handle format descriptors
- Test output to stdout
**Verification**: `program hello; print *, "Hello, World!"; end program` prints correctly

---

## Phase 3: Variables (Week 3)

### Issue #9: Implement hlfir.declare [8 points]
**Goal**: Variable declarations via HLFIR
**Depends on**: #3
**Tasks**:
- Create hlfir.declare operation builder
- Map Fortran types to MLIR types
- Handle variable attributes
- Generate proper debug info
**Verification**: Variables declared in HLFIR

### Issue #10: Integer variables [5 points]
**Goal**: Support integer type
**Depends on**: #9
**Tasks**:
- Map INTEGER to i32/i64
- Handle integer literals via arith.constant
- Support initialization
- Test integer operations
**Verification**: `integer :: i = 42` works

### Issue #11: Real variables [5 points]
**Goal**: Support real type
**Depends on**: #9
**Tasks**:
- Map REAL to f32/f64
- Handle real literals via arith.constant
- Support initialization
- Test real operations
**Verification**: `real :: x = 3.14` works

### Issue #12: Character variables [5 points]
**Goal**: Support character type
**Depends on**: #9
**Tasks**:
- Map CHARACTER to !fir.char
- Handle string length
- Support initialization
- Test character operations
**Verification**: `character(len=10) :: name = "test"` works

---

## Phase 4: Operations (Week 4)

### Issue #13: Binary arithmetic [8 points]
**Goal**: Support +, -, *, /
**Depends on**: #10, #11
**Tasks**:
- Generate arith.addi/addf operations
- Generate arith.subi/subf operations
- Generate arith.muli/mulf operations
- Generate arith.divi/divf operations
**Verification**: `x = 2 + 3 * 4` compiles

### Issue #14: Implement hlfir.assign [8 points]
**Goal**: Assignment statements
**Depends on**: #9
**Tasks**:
- Create hlfir.assign builder
- Handle scalar assignments
- Support type conversion in assign
- Test various assignments
**Verification**: `x = y + 1` works

### Issue #15: Comparison operations [5 points]
**Goal**: Support relational operators
**Depends on**: #10
**Tasks**:
- Generate arith.cmpi/cmpf operations
- Map Fortran comparisons to MLIR
- Return i1 (logical) results
- Test all comparison ops
**Verification**: `x > 0` evaluates correctly

---

## Phase 5: Control Flow (Week 5)

### Issue #16: If-then statements [8 points]
**Goal**: Conditional execution
**Depends on**: #15
**Tasks**:
- Generate scf.if operations
- Create then/else regions
- Handle SSA values across blocks
- Test nested if statements
**Verification**: `if (x > 0) print *, "positive"` works

### Issue #17: Do loops [8 points]
**Goal**: Iterative loops
**Depends on**: #10
**Tasks**:
- Generate scf.for operations
- Handle loop variables
- Support loop bounds
- Test nested loops
**Verification**: `do i = 1, 10; print *, i; end do` works

### Issue #18: Do while loops [5 points]
**Goal**: Conditional loops
**Depends on**: #15
**Tasks**:
- Generate scf.while operations
- Handle loop conditions
- Test with various conditions
**Verification**: `do while (x > 0)` works

---

## Phase 6: Arrays (Week 6-7)

### Issue #19: Fixed-size arrays [8 points]
**Goal**: Static array declarations
**Depends on**: #9
**Tasks**:
- Generate !fir.array types
- Create array allocations
- Initialize arrays
- Test multidimensional arrays
**Verification**: `integer :: arr(10)` works

### Issue #20: hlfir.designate for indexing [8 points]
**Goal**: Array element access
**Depends on**: #19
**Tasks**:
- Create hlfir.designate builder
- Handle array subscripts
- Support multidimensional access
- Test bounds checking
**Verification**: `arr(i) = 42` works

### Issue #21: hlfir.elemental operations [8 points]
**Goal**: Array expressions
**Depends on**: #19
**Tasks**:
- Create hlfir.elemental builder
- Generate element-wise operations
- Handle array temporaries
- Test array arithmetic
**Verification**: `arr = arr + 1` works

### Issue #22: Array sections [8 points]
**Goal**: Array slicing
**Depends on**: #20
**Tasks**:
- Generate bounds for designate
- Handle stride specification
- Support partial sections
- Test various slices
**Verification**: `arr(1:5) = 0` works

---

## Phase 7: Procedures (Week 8)

### Issue #23: Subroutine definitions [8 points]
**Goal**: Compile subroutines
**Depends on**: #3
**Tasks**:
- Generate func.func for subroutines
- Handle parameter lists
- Support local variables
- Test call/return
**Verification**: `subroutine mysub()` compiles

### Issue #24: Subroutine calls [5 points]
**Goal**: Call statements
**Depends on**: #23
**Tasks**:
- Generate func.call operations
- Handle argument passing
- Support by-reference semantics
- Test with various arguments
**Verification**: `call mysub()` works

### Issue #25: Function definitions [8 points]
**Goal**: Compile functions
**Depends on**: #23
**Tasks**:
- Handle result variables
- Generate func.return operations
- Support function types
- Test return values
**Verification**: `function myfunc()` compiles

### Issue #26: Function calls [5 points]
**Goal**: Function invocation
**Depends on**: #25
**Tasks**:
- Generate function calls
- Handle return values
- Use in expressions
- Test various returns
**Verification**: `x = myfunc()` works

---

## Phase 8: Intrinsics (Week 9)

### Issue #27: Math intrinsics [8 points]
**Goal**: sin, cos, sqrt, etc.
**Depends on**: #26
**Tasks**:
- Identify intrinsic functions
- Map to LLVM intrinsics or runtime
- Handle type-specific versions
- Test accuracy
**Verification**: `x = sin(y)` works

### Issue #28: Min/max intrinsics [5 points]
**Goal**: min/max functions
**Depends on**: #13
**Tasks**:
- Generate select operations
- Handle multiple arguments
- Support type variants
- Test edge cases
**Verification**: `x = max(a, b, c)` works

### Issue #29: Array intrinsics [8 points]
**Goal**: sum, product, etc.
**Depends on**: #21
**Tasks**:
- Generate reduction operations
- Handle array arguments
- Support dimensional reduction
- Test performance
**Verification**: `total = sum(arr)` works

---

## Phase 9: Dynamic Memory (Week 10)

### Issue #30: Allocatable arrays [13 points]
**Goal**: Dynamic arrays
**Depends on**: #19
**Tasks**:
- Generate !fir.box types
- Create descriptors
- Handle metadata
- Test allocation patterns
**Verification**: `allocatable :: arr(:)` works

### Issue #31: Allocate statements [8 points]
**Goal**: Memory allocation
**Depends on**: #30
**Tasks**:
- Generate fir.allocmem operations
- Update descriptors
- Handle allocation errors
- Test various sizes
**Verification**: `allocate(arr(n))` works

### Issue #32: Deallocate statements [5 points]
**Goal**: Memory deallocation
**Depends on**: #31
**Tasks**:
- Generate fir.freemem operations
- Clear descriptors
- Handle deallocation errors
- Test memory leaks
**Verification**: `deallocate(arr)` works

---

## Phase 10: Derived Types (Week 11)

### Issue #33: Type definitions [8 points]
**Goal**: User-defined types
**Depends on**: #9
**Tasks**:
- Generate !fir.type definitions
- Map components to fields
- Handle type names
- Test nested types
**Verification**: `type :: point` works

### Issue #34: Type variables [5 points]
**Goal**: Derived type instances
**Depends on**: #33
**Tasks**:
- Declare type variables
- Handle initialization
- Support in procedures
- Test assignment
**Verification**: `type(point) :: p` works

### Issue #35: Component access [5 points]
**Goal**: Field access
**Depends on**: #34
**Tasks**:
- Generate hlfir.designate for components
- Handle nested access
- Support assignment
- Test all types
**Verification**: `p%x = 1.0` works

---

## Phase 11: Modules (Week 12)

### Issue #36: Module compilation [13 points]
**Goal**: Compile modules
**Depends on**: #3
**Tasks**:
- Generate module structure
- Export symbols
- Create .mod files
- Test visibility
**Verification**: `module mymod` compiles

### Issue #37: Use statements [8 points]
**Goal**: Import modules
**Depends on**: #36
**Tasks**:
- Read .mod files
- Import symbols
- Handle renaming
- Test dependencies
**Verification**: `use mymod` works

---

## Phase 12: Lazy Fortran (Week 13)

### Issue #38: .lf file support [3 points]
**Goal**: Handle lazy Fortran files
**Depends on**: fortfront support
**Tasks**:
- Recognize .lf extension
- Enable inference mode
- Pass to fortfront
- Test basic .lf
**Verification**: ffc accepts .lf files

### Issue #39: Implicit declarations [8 points]
**Goal**: Type inference support
**Depends on**: #38, #9
**Tasks**:
- Get inferred types from fortfront
- Generate hlfir.declare automatically
- Handle all basic types
- Test inference
**Verification**: `x = 2` infers integer

### Issue #40: Implicit program [5 points]
**Goal**: Auto-wrap statements
**Depends on**: #38
**Tasks**:
- Detect missing program
- Add program wrapper
- Handle implicit none
- Test bare statements
**Verification**: Bare statements compile

---

## Phase 13: Optimization (Week 14)

### Issue #41: Basic optimizations [8 points]
**Goal**: -O1/-O2 support
**Depends on**: #4
**Tasks**:
- Add optimization flags
- Configure MLIR passes
- Enable LLVM optimizations
- Benchmark performance
**Verification**: -O2 produces faster code

### Issue #42: Link-time optimization [5 points]
**Goal**: LTO support
**Depends on**: #41
**Tasks**:
- Enable LTO in LLVM
- Configure thin LTO
- Test with large programs
- Measure improvements
**Verification**: --lto works

---

## Phase 14: Polish (Week 15)

### Issue #43: Error messages [5 points]
**Goal**: Better diagnostics
**Depends on**: fortfront
**Tasks**:
- Add source locations
- Improve error text
- Add fix suggestions
- Test error cases
**Verification**: Clear errors with line numbers

### Issue #44: CLI compatibility [5 points]
**Goal**: Match gfortran interface
**Tasks**:
- Support common flags
- Add help text
- Handle multiple files
- Test compatibility
**Verification**: Drop-in replacement

### Issue #45: Documentation [3 points]
**Goal**: User documentation
**Tasks**:
- Write user guide
- Document all flags
- Add examples
- Create tutorials
**Verification**: Complete docs

---

## Cleanup Tasks (Ongoing)

### Issue #46: Remove text-based MLIR [5 points]
**Goal**: Clean up legacy code
**When**: After Phase 1 complete
**Tasks**:
- Delete all string concatenation
- Remove text generation modules
- Update all tests
- Verify no text remains
**Verification**: No MLIR text generation

### Issue #47: Remove stub implementations [3 points]
**Goal**: Replace all stubs
**When**: After relevant phases
**Tasks**:
- Replace C stub functions
- Remove mock implementations
- Verify all real
**Verification**: No stubs remain

---

## Success Metrics

### Week 1: Foundation
- ✅ Empty program compiles
- ✅ Executable runs
- ✅ LLVM IR visible

### Week 2: Hello World
- ✅ Print works
- ✅ Strings supported
- ✅ Runtime linked

### Week 4: Basic Programs
- ✅ Variables work
- ✅ Arithmetic works
- ✅ Assignment works

### Week 6: Real Programs
- ✅ Control flow works
- ✅ Arrays work
- ✅ Loops work

### Week 8: Modular Code
- ✅ Procedures work
- ✅ Functions work
- ✅ Calls work

### Week 13: Production Ready
- ✅ All features work
- ✅ Lazy Fortran works
- ✅ Performance good

### Week 15: Release
- ✅ Fully documented
- ✅ All tests pass
- ✅ Users happy

---

## Risk Mitigation

### Technical Risks
1. **fortfront blocking**: Use mock bridge initially; fpm handles linking automatically
2. **MLIR API changes**: Pin LLVM version
3. **Runtime issues**: Test with multiple runtimes
4. **Performance problems**: Profile early
5. **Integration complexity**: **MITIGATED** - fpm automatic dependency management simplifies fortfront integration

### Process Risks
1. **Scope creep**: Strict phase adherence
2. **Complex bugs**: Incremental testing
3. **Integration issues**: Test each component
4. **Documentation lag**: Update with code

---

## Notes

- **Points**: Fibonacci scale complexity (1,2,3,5,8,13)
- **Dependencies**: Must complete in order shown
- **Atomic**: Each issue leaves system working
- **HLFIR-first**: Always prefer HLFIR operations
- **Testing**: Every issue includes tests