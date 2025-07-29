program test_associated_intrinsic
    use ast_core
    use ast_factory
    use backend_interface
    use backend_factory
    use temp_utils
    implicit none

    logical :: all_passed = .true.

    print *, "=== Testing Associated Intrinsic ==="
    print *, ""

    all_passed = all_passed .and. test_associated_null_pointer()
    all_passed = all_passed .and. test_associated_valid_pointer()
    all_passed = all_passed .and. test_associated_target_comparison()

    if (all_passed) then
        print *, ""
        print *, "All associated intrinsic tests passed!"
        stop 0
    else
        print *, ""
        print *, "Some associated intrinsic tests failed!"
        stop 1
    end if

contains

    function test_associated_null_pointer() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, ptr_decl_idx, call_idx, print_idx
        integer :: ptr_id_idx, no_args(0), result_id_idx

        print *, "Testing associated(null_pointer)..."

        passed = .false.

        ! Create AST: integer, pointer :: ptr
        ! print *, associated(ptr)
        ptr_decl_idx = push_declaration(arena, "integer", "ptr", is_pointer=.true.)

        ! Create identifier for pointer
        ptr_id_idx = push_identifier(arena, "ptr")

        ! Create associated intrinsic call: associated(ptr)
        no_args = [integer ::]
        call_idx = push_call_or_subscript(arena, "associated", [ptr_id_idx])

        ! Create result identifier for print
        result_id_idx = push_identifier(arena, "result")

        ! Create print statement with associated call
        print_idx = push_print_statement(arena, "*", [call_idx])

        prog_idx = push_program(arena, "test_assoc_null", [ptr_decl_idx, print_idx])

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

        ! Check if associated intrinsic support exists
        if (index(mlir_code, "llvm.icmp") > 0 .and. &
            (index(mlir_code, "ne") > 0 .or. index(mlir_code, "eq") > 0)) then
            print *, "PASS: Associated intrinsic generates proper MLIR"
            print *, "  Generated pointer comparison operation"
            passed = .true.
        else
            print *, "FAIL: No associated intrinsic support"
            print *, "  Expected: LLVM icmp operation for null pointer check"
            print *, "  Got length:", len(mlir_code)
            if (len(mlir_code) > 0) then
                print *, "  First 500 chars: ", mlir_code(1:min(500, len(mlir_code)))
            else
                print *, "  Empty MLIR!"
            end if
            passed = .false.
        end if
    end function test_associated_null_pointer

    function test_associated_valid_pointer() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
     integer :: prog_idx, ptr_decl_idx, target_decl_idx, assign_idx, call_idx, print_idx
        integer :: ptr_id_idx, target_id_idx

        print *, "Testing associated(valid_pointer)..."

        passed = .false.

        ! Create AST:
        ! integer, pointer :: ptr
        ! integer, target :: x
        ! ptr => x
        ! print *, associated(ptr)
        ptr_decl_idx = push_declaration(arena, "integer", "ptr", is_pointer=.true.)
        target_decl_idx = push_declaration(arena, "integer", "x")

        ! Create identifiers
        ptr_id_idx = push_identifier(arena, "ptr")
        target_id_idx = push_identifier(arena, "x")

        ! Create pointer assignment: ptr => x
        assign_idx = push_pointer_assignment(arena, ptr_id_idx, target_id_idx)

        ! Create associated intrinsic call: associated(ptr)
        call_idx = push_call_or_subscript(arena, "associated", [ptr_id_idx])

        ! Create print statement
        print_idx = push_print_statement(arena, "*", [call_idx])

        prog_idx = push_program(arena, "test_assoc_valid", &
                                [ptr_decl_idx, target_decl_idx, assign_idx, print_idx])

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

        ! Check if associated intrinsic works with valid pointer
        if (index(mlir_code, "llvm.icmp") > 0 .and. &
      (index(mlir_code, "llvm.load") > 0 .or. index(mlir_code, "memref.load") > 0)) then
            print *, "PASS: Associated with valid pointer generates proper MLIR"
            print *, "  Generated load and comparison operations"
            passed = .true.
        else
            print *, "FAIL: No associated intrinsic support for valid pointer"
            print *, "  Expected: LLVM load/memref.load and icmp operations"
            print *, "  Got: ", trim(mlir_code)
            passed = .false.
        end if
    end function test_associated_valid_pointer

    function test_associated_target_comparison() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, ptr_decl_idx, target1_decl_idx, target2_decl_idx
        integer :: assign_idx, call_idx, print_idx
        integer :: ptr_id_idx, target1_id_idx, target2_id_idx

        print *, "Testing associated(pointer, target)..."

        passed = .false.

        ! Create AST:
        ! integer, pointer :: ptr
        ! integer, target :: x, y
        ! ptr => x
        ! print *, associated(ptr, x)
        ptr_decl_idx = push_declaration(arena, "integer", "ptr", is_pointer=.true.)
        target1_decl_idx = push_declaration(arena, "integer", "x")
        target2_decl_idx = push_declaration(arena, "integer", "y")

        ! Create identifiers
        ptr_id_idx = push_identifier(arena, "ptr")
        target1_id_idx = push_identifier(arena, "x")
        target2_id_idx = push_identifier(arena, "y")

        ! Create pointer assignment: ptr => x
        assign_idx = push_pointer_assignment(arena, ptr_id_idx, target1_id_idx)

        ! Create associated intrinsic call: associated(ptr, x)
    call_idx = push_call_or_subscript(arena, "associated", [ptr_id_idx, target1_id_idx])

        ! Create print statement
        print_idx = push_print_statement(arena, "*", [call_idx])

        prog_idx = push_program(arena, "test_assoc_target", &
                                [ptr_decl_idx, target1_decl_idx, target2_decl_idx, &
                                 assign_idx, print_idx])

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

        ! Check if associated intrinsic works with target comparison
        if (index(mlir_code, "llvm.icmp") > 0 .and. &
            index(mlir_code, "memref.extract_aligned_pointer") > 0) then
            print *, "PASS: Associated with target comparison generates proper MLIR"
            print *, "  Generated address extraction and comparison"
            passed = .true.
        else
            print *, "FAIL: No associated intrinsic support for target comparison"
            print *, "  Expected: Address extraction and icmp operations"
            print *, "  Got: ", trim(mlir_code)
            passed = .false.
        end if
    end function test_associated_target_comparison

end program test_associated_intrinsic
