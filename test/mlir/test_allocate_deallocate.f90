program test_allocate_deallocate
    use ast_core
    use ast_factory
    use backend_interface
    use backend_factory
    use temp_utils
    implicit none

    logical :: all_passed = .true.

    print *, "=== Testing Allocate/Deallocate Statements ==="
    print *, ""

    all_passed = all_passed .and. test_simple_allocate()
    all_passed = all_passed .and. test_array_allocate()
    all_passed = all_passed .and. test_deallocate()
    all_passed = all_passed .and. test_allocate_with_stat()

    if (all_passed) then
        print *, ""
        print *, "All allocate/deallocate tests passed!"
        stop 0
    else
        print *, ""
        print *, "Some allocate/deallocate tests failed!"
        stop 1
    end if

contains

    function test_simple_allocate() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, decl_idx, alloc_idx, id_idx

        print *, "Testing simple allocate statement..."

        passed = .false.

        ! Create AST with pointer declaration and allocate statement
        decl_idx = push_declaration(arena, "integer", "ptr", is_pointer=.true.)

        ! Create an identifier node for the variable to allocate
        ! First create identifier that references the declared variable
        id_idx = push_identifier(arena, "ptr")

        ! Now create allocate statement with the identifier
        alloc_idx = push_allocate(arena, [id_idx])

        prog_idx = push_program(arena, "test_allocate", [decl_idx, alloc_idx])

        ! Configure backend for compile mode
        backend_opts%compile_mode = .true.

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR code
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, error_msg)

        ! Check if allocate support exists (looking for dynamic allocation)
      if (index(mlir_code, "malloc") > 0 .or. index(mlir_code, "memref.alloc") > 0) then
            print *, "PASS: Dynamic allocate statement support exists"
            print *, "  Generated MLIR includes dynamic allocation"
            passed = .true.
        else
            print *, "FAIL: No dynamic allocate statement support"
            print *, "  Expected: Dynamic memory allocation (malloc/memref.alloc)"
            print *, "  Got: ", trim(mlir_code)
            passed = .false.
        end if
    end function test_simple_allocate

    function test_array_allocate() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, decl_idx, dim_idx, alloc_idx

        print *, "Testing array allocate statement..."

        passed = .false.

        ! Create AST with allocatable array declaration
        decl_idx = push_declaration(arena, "real", "array", &
                                    is_allocatable=.true.)

        ! Try to create allocate statement with dimensions
        ! allocate_idx = push_allocate(arena, "array", dimensions=[10,20])
        alloc_idx = 0  ! Placeholder

        prog_idx = push_program(arena, "test_array_allocate", [decl_idx])

        ! Configure backend for compile mode
        backend_opts%compile_mode = .true.

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR code
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, error_msg)

        ! Check current behavior
        print *, "CURRENT BEHAVIOR: No allocate statement in AST"
        print *, "  Expected: Dynamic array allocation with runtime dimensions"
        print *, "  Got: Only static allocations possible"
        passed = .false.  ! This demonstrates the limitation
    end function test_array_allocate

    function test_deallocate() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, decl_idx, dealloc_idx, id_idx, alloc_idx

        print *, "Testing deallocate statement..."

        passed = .false.

        ! Create AST with pointer/allocatable declaration
        decl_idx = push_declaration(arena, "character", "str", &
                                    is_pointer=.true.)

        ! Create identifier for allocate and deallocate
        id_idx = push_identifier(arena, "str")

        ! First allocate, then deallocate
        alloc_idx = push_allocate(arena, [id_idx])
        dealloc_idx = push_deallocate(arena, [id_idx])

   prog_idx = push_program(arena, "test_deallocate", [decl_idx, alloc_idx, dealloc_idx])

        ! Configure backend for compile mode
        backend_opts%compile_mode = .true.

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR code
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, error_msg)

        ! Check if deallocate support exists
      if (index(mlir_code, "free") > 0 .or. index(mlir_code, "memref.dealloc") > 0) then
            print *, "PASS: Deallocate statement support exists"
            print *, "  Generated MLIR includes memory deallocation"
            passed = .true.
        else
            print *, "FAIL: No deallocate statement support"
            print *, "  Expected: Memory deallocation (free/memref.dealloc)"
            print *, "  Got: ", trim(mlir_code)
            passed = .false.
        end if
    end function test_deallocate

    function test_allocate_with_stat() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, ptr_idx, stat_idx, alloc_idx

        print *, "Testing allocate with stat parameter..."

        passed = .false.

        ! Create AST with pointer and status variable
        ptr_idx = push_declaration(arena, "double precision", "data", &
                                   is_pointer=.true.)
        stat_idx = push_declaration(arena, "integer", "status")

        ! Try to create allocate with stat
        ! allocate_idx = push_allocate(arena, "data", stat="status")
        alloc_idx = 0  ! Placeholder

        prog_idx = push_program(arena, "test_allocate_stat", [ptr_idx, stat_idx])

        ! Configure backend for compile mode
        backend_opts%compile_mode = .true.

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR code
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, error_msg)

        ! This test demonstrates the limitation
        print *, "EXPECTED: No allocate with stat support"
        print *, "  Allocate statement nodes don't exist in AST"
        passed = .false.
    end function test_allocate_with_stat

end program test_allocate_deallocate
