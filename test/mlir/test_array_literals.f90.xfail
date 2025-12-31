program test_array_literals
    use iso_fortran_env, only: error_unit
    use mlir_backend
    use backend_factory
    use backend_interface
    use fortfront, only: ast_arena_t, create_ast_arena, LITERAL_INTEGER, LITERAL_REAL
    use ast_factory
    use mlir_utils, only: int_to_str
    implicit none

    logical :: all_tests_passed

    print *, "=== Array Literal Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all array literal tests
    if (.not. test_integer_array_literal()) all_tests_passed = .false.
    if (.not. test_array_literal_in_assignment()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All array literal tests passed!"
        stop 0
    else
        print *, "Some array literal tests failed!"
        stop 1
    end if

contains

    function test_integer_array_literal() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: arr_literal_idx, assign_idx, prog_idx
        integer :: elem_indices(5), i

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: [1, 2, 3, 4, 5]
        arena = create_ast_arena()

        ! Create element nodes first
        do i = 1, 5
            elem_indices(i) = push_literal(arena, int_to_str(i), LITERAL_INTEGER)
        end do

        ! Create array literal
        arr_literal_idx = push_array_literal(arena, elem_indices)

        ! Create identifier for variable
        elem_indices(1) = push_identifier(arena, "x")

        ! Assign to variable: x = [1, 2, 3, 4, 5]
        assign_idx = push_assignment(arena, elem_indices(1), arr_literal_idx)
        prog_idx = push_program(arena, "test", [assign_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper array literal generation
            if (index(output, "arith.constant dense<[1, 2, 3, 4, 5]>") > 0 .or. &
                index(output, "memref.alloca") > 0 .and. &
                index(output, "memref.store") > 0) then
                print *, "PASS: Integer array literal generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper array literal generation"
                print *, "Expected: dense constant or memref operations"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_integer_array_literal

    function test_array_literal_in_assignment() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: decl_idx, arr_literal_idx, assign_idx, prog_idx, arr_id_idx
        integer :: elem_indices(5), i, dim_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: integer :: arr(5)
        !       arr = [10, 20, 30, 40, 50]
        arena = create_ast_arena()

        ! Declare array - need to push dimension as literal nodes
        dim_idx = push_literal(arena, "5", LITERAL_INTEGER)
       decl_idx = push_declaration(arena, "integer", ["arr"], dimension_indices=[dim_idx])

        ! Create element nodes for array literal
        do i = 1, 5
            elem_indices(i) = push_literal(arena, int_to_str(i*10), LITERAL_INTEGER)
        end do

        ! Create array literal
        arr_literal_idx = push_array_literal(arena, elem_indices)

        ! Create identifier for array variable
        arr_id_idx = push_identifier(arena, "arr")

        ! Assign literal to array
        assign_idx = push_assignment(arena, arr_id_idx, arr_literal_idx)
        prog_idx = push_program(arena, "test", [decl_idx, assign_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper array assignment - looking for memref.copy
if ((index(output, "memref.alloca") > 0 .or. index(output, "memref.global") > 0) .and. &
                (index(output, "memref.copy") > 0 .or. &
                 index(output, "memref.store") > 0)) then
                print *, "PASS: Array literal assignment generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper array assignment operations"
                print *, "Expected: memref operations for array assignment"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_array_literal_in_assignment

end program test_array_literals
