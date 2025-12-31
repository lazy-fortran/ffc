program test_pointer_assignment
    use ast_core
    use ast_factory
    use backend_interface
    use backend_factory
    use temp_utils
    implicit none

    logical :: all_passed = .true.

    print *, "=== Testing Pointer Assignment ==="
    print *, ""

    all_passed = all_passed .and. test_simple_pointer_assignment()
    all_passed = all_passed .and. test_pointer_null_assignment()
    all_passed = all_passed .and. test_pointer_to_pointer_assignment()

    if (all_passed) then
        print *, ""
        print *, "All pointer assignment tests passed!"
        stop 0
    else
        print *, ""
        print *, "Some pointer assignment tests failed!"
        stop 1
    end if

contains

    function test_simple_pointer_assignment() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, ptr_decl_idx, target_decl_idx, assign_idx
        integer :: ptr_id_idx, target_id_idx

        print *, "Testing simple pointer assignment..."

        passed = .false.

        ! Create AST with pointer and regular declarations for now
        ! integer, pointer :: ptr
        ! integer :: x
        ! ptr => x (regular assignment for demonstration)
        ptr_decl_idx = push_declaration(arena, "integer", "ptr", is_pointer=.true.)
        target_decl_idx = push_declaration(arena, "integer", "x")

        ! Create identifiers for assignment
        ptr_id_idx = push_identifier(arena, "ptr")
        target_id_idx = push_identifier(arena, "x")

        ! Create pointer assignment statement (ptr => x)
        assign_idx = push_pointer_assignment(arena, ptr_id_idx, target_id_idx)

        prog_idx = push_program(arena, "test_ptr_assign", &
                                [ptr_decl_idx, target_decl_idx, assign_idx])

        ! Configure backend for compile mode
        backend_opts%compile_mode = .true.

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR code
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, &
                                   error_msg)

        ! Check if pointer assignment support exists
        if (index(mlir_code, "llvm.store") > 0 .and. &
            index(mlir_code, "memref.extract") > 0) then
            print *, "PASS: Pointer assignment generates proper MLIR"
            print *, "  Generated pointer store operation"
            passed = .true.
        else
            print *, "FAIL: No pointer assignment support"
            print *, "  Expected: LLVM store with address extraction"
            print *, "  Got: ", trim(mlir_code)
            passed = .false.
        end if
    end function test_simple_pointer_assignment

    function test_pointer_null_assignment() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, ptr_decl_idx, assign_idx
        integer :: ptr_id_idx, null_idx
        integer, parameter :: no_args(0) = [integer ::]

        print *, "Testing pointer null assignment..."

        passed = .false.

        ! Create AST with pointer declaration
        ! integer, pointer :: ptr
        ! ptr => null()
        ptr_decl_idx = push_declaration(arena, "integer", "ptr", is_pointer=.true.)

        ! Create identifier and null intrinsic
        ptr_id_idx = push_identifier(arena, "ptr")
        null_idx = push_call_or_subscript(arena, "null", no_args)

        ! Create pointer assignment to null (ptr => null())
        assign_idx = push_pointer_assignment(arena, ptr_id_idx, null_idx)

        prog_idx = push_program(arena, "test_ptr_null", &
                                [ptr_decl_idx, assign_idx])

        ! Configure backend for compile mode
        backend_opts%compile_mode = .true.

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR code
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, &
                                   error_msg)

        ! Check if null assignment support exists
        if (index(mlir_code, "llvm.mlir.null") > 0 .or. &
            index(mlir_code, "llvm.mlir.zero") > 0) then
            print *, "PASS: Null pointer assignment generates proper MLIR"
            print *, "  Generated null pointer constant"
            passed = .true.
        else
            print *, "FAIL: No null pointer assignment support"
            print *, "  Expected: LLVM null/zero constant"
            print *, "  Got: ", trim(mlir_code)
            passed = .false.
        end if
    end function test_pointer_null_assignment

    function test_pointer_to_pointer_assignment() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_opts
        character(len=:), allocatable :: mlir_code
        character(len=256) :: error_msg
        integer :: prog_idx, ptr1_decl_idx, ptr2_decl_idx, assign_idx
        integer :: ptr1_id_idx, ptr2_id_idx

        print *, "Testing pointer to pointer assignment..."

        passed = .false.

        ! Create AST with two pointer declarations
        ! integer, pointer :: ptr1, ptr2
        ! ptr1 => ptr2
        ptr1_decl_idx = push_declaration(arena, "integer", "ptr1", &
                                         is_pointer=.true.)
        ptr2_decl_idx = push_declaration(arena, "integer", "ptr2", &
                                         is_pointer=.true.)

        ! Create identifiers for assignment
        ptr1_id_idx = push_identifier(arena, "ptr1")
        ptr2_id_idx = push_identifier(arena, "ptr2")

        ! Create pointer assignment statement (ptr1 => ptr2)
        assign_idx = push_pointer_assignment(arena, ptr1_id_idx, ptr2_id_idx)

        prog_idx = push_program(arena, "test_ptr_to_ptr", &
                                [ptr1_decl_idx, ptr2_decl_idx, assign_idx])

        ! Configure backend for compile mode
        backend_opts%compile_mode = .true.

        ! Create MLIR backend
        call create_backend("mlir", backend, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Error creating backend:", trim(error_msg)
            return
        end if

        ! Generate MLIR code
        call backend%generate_code(arena, prog_idx, backend_opts, mlir_code, &
                                   error_msg)

        ! Check if pointer to pointer assignment exists
        if (index(mlir_code, "llvm.load") > 0 .and. &
            index(mlir_code, "llvm.store") > 0) then
            print *, "PASS: Pointer to pointer assignment generates proper MLIR"
            print *, "  Generated load/store operations"
            passed = .true.
        else
            print *, "FAIL: No pointer to pointer assignment support"
            print *, "  Expected: LLVM load/store operations"
            print *, "  Got: ", trim(mlir_code)
            passed = .false.
        end if
    end function test_pointer_to_pointer_assignment

end program test_pointer_assignment
