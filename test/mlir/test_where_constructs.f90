program test_where_constructs
    use iso_fortran_env, only: error_unit
    use mlir_backend
    use backend_factory
    use backend_interface
    use ast_core, only: ast_arena_t, create_ast_stack, LITERAL_INTEGER, LITERAL_REAL
    use ast_factory
    use mlir_utils, only: int_to_str
    implicit none

    logical :: all_tests_passed

    print *, "=== Where Construct Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all where construct tests
    if (.not. test_simple_where()) all_tests_passed = .false.
    if (.not. test_where_with_elsewhere()) all_tests_passed = .false.
    if (.not. test_nested_where()) all_tests_passed = .false.
    if (.not. test_where_array_assignment()) all_tests_passed = .false.
    if (.not. test_where_with_mask_expression()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All where construct tests passed!"
        stop 0
    else
        print *, "Some where construct tests failed!"
        stop 1
    end if

contains

    function test_simple_where() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: where_idx, prog_idx, mask_idx, a_idx, b_idx, assign_idx
        integer :: zero_id, gt_op_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: where (a > 0)
        !         b = a
        !       end where
        arena = create_ast_stack()

        ! Create mask expression: a > 0
        a_idx = push_identifier(arena, "a")
        zero_id = push_literal(arena, "0", LITERAL_INTEGER)
        gt_op_idx = push_binary_op(arena, a_idx, zero_id, ">")

        ! Create assignment: b = a
        b_idx = push_identifier(arena, "b")
        assign_idx = push_assignment(arena, b_idx, a_idx)

        ! Create where construct
        where_idx = push_where_construct(arena, gt_op_idx, [assign_idx])
        prog_idx = push_program(arena, "test", [where_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper where generation
            if ((index(output, "scf.if") > 0 .or. &
                 index(output, "scf.parallel") > 0) .and. &
                (index(output, "where") > 0 .or. &
                 index(output, "mask") > 0 .or. &
                 index(output, "memref.load") > 0)) then
                print *, "PASS: Simple where construct generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper where construct generation"
                print *, "Expected: conditional with masked array operations"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_simple_where

    function test_where_with_elsewhere() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: where_idx, prog_idx, mask_idx, assign1_idx, assign2_idx
        integer :: a_idx, b_idx, c_idx, zero_id, gt_op_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: where (a > 0)
        !         b = a
        !       elsewhere
        !         b = c
        !       end where
        arena = create_ast_stack()

        ! Create mask expression: a > 0
        a_idx = push_identifier(arena, "a")
        zero_id = push_literal(arena, "0", LITERAL_INTEGER)
        gt_op_idx = push_binary_op(arena, a_idx, zero_id, ">")

        ! Create assignments
        b_idx = push_identifier(arena, "b")
        c_idx = push_identifier(arena, "c")
        assign1_idx = push_assignment(arena, b_idx, a_idx)
        assign2_idx = push_assignment(arena, b_idx, c_idx)

        ! Create where construct with elsewhere
        where_idx = push_where_construct_with_elsewhere(arena, gt_op_idx, [assign1_idx], [assign2_idx])
        prog_idx = push_program(arena, "test", [where_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper where/elsewhere generation
            if ((index(output, "scf.if") > 0) .and. &
                (index(output, "else") > 0 .or. &
                 index(output, "elsewhere") > 0) .and. &
                (index(output, "where") > 0 .or. &
                 index(output, "mask") > 0)) then
                print *, "PASS: Where with elsewhere generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper where/elsewhere generation"
                print *, "Expected: conditional with else block and masking"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_where_with_elsewhere

    function test_nested_where() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: outer_where_idx, inner_where_idx, prog_idx
        integer :: mask1_idx, mask2_idx, assign_idx
        integer :: a_idx, b_idx, c_idx, zero_id, ten_id, gt_op1_idx, lt_op_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: where (a > 0)
        !         where (a < 10)
        !           b = a
        !         end where
        !       end where
        arena = create_ast_stack()

        ! Create mask expressions
        a_idx = push_identifier(arena, "a")
        zero_id = push_literal(arena, "0", LITERAL_INTEGER)
        ten_id = push_literal(arena, "10", LITERAL_INTEGER)
        gt_op1_idx = push_binary_op(arena, a_idx, zero_id, ">")
        lt_op_idx = push_binary_op(arena, a_idx, ten_id, "<")

        ! Create assignment: b = a
        b_idx = push_identifier(arena, "b")
        assign_idx = push_assignment(arena, b_idx, a_idx)

        ! Create inner where
        inner_where_idx = push_where_construct(arena, lt_op_idx, [assign_idx])

        ! Create outer where
        outer_where_idx = push_where_construct(arena, gt_op1_idx, [inner_where_idx])
        prog_idx = push_program(arena, "test", [outer_where_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper nested where generation
            if ((index(output, "scf.if") > 0) .and. &
                (index(output, "where") > 0 .or. &
                 count_occurrences(output, "scf.if") >= 2)) then
                print *, "PASS: Nested where constructs generate proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper nested where generation"
                print *, "Expected: nested conditionals with masking"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_nested_where

    function test_where_array_assignment() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: where_idx, prog_idx, mask_idx, assign_idx
        integer :: a_ref_idx, b_ref_idx, c_ref_idx, i_idx, zero_id, gt_op_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: where (a(i) > 0)
        !         b(i) = c(i)
        !       end where
        arena = create_ast_stack()

        ! Create array references
        i_idx = push_identifier(arena, "i")
        a_ref_idx = push_call_or_subscript(arena, "a", [i_idx])
        b_ref_idx = push_call_or_subscript(arena, "b", [i_idx])
        c_ref_idx = push_call_or_subscript(arena, "c", [i_idx])

        ! Create mask expression: a(i) > 0
        zero_id = push_literal(arena, "0", LITERAL_INTEGER)
        gt_op_idx = push_binary_op(arena, a_ref_idx, zero_id, ">")

        ! Create assignment: b(i) = c(i)
        assign_idx = push_assignment(arena, b_ref_idx, c_ref_idx)

        ! Create where construct
        where_idx = push_where_construct(arena, gt_op_idx, [assign_idx])
        prog_idx = push_program(arena, "test", [where_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper array where generation
            if ((index(output, "memref.load") > 0 .and. &
                 index(output, "memref.store") > 0) .and. &
                (index(output, "scf.if") > 0 .or. &
                 index(output, "scf.parallel") > 0)) then
                print *, "PASS: Where with array assignment generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper array where generation"
                print *, "Expected: memory operations with conditional masking"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_where_array_assignment

    function test_where_with_mask_expression() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: where_idx, prog_idx, mask_idx, assign_idx
        integer :: a_idx, b_idx, c_idx, zero_id, ten_id
        integer :: gt_op_idx, lt_op_idx, and_op_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: where (a > 0 .and. a < 10)
        !         b = c
        !       end where
        arena = create_ast_stack()

        ! Create complex mask expression: a > 0 .and. a < 10
        a_idx = push_identifier(arena, "a")
        zero_id = push_literal(arena, "0", LITERAL_INTEGER)
        ten_id = push_literal(arena, "10", LITERAL_INTEGER)
        gt_op_idx = push_binary_op(arena, a_idx, zero_id, ">")
        lt_op_idx = push_binary_op(arena, a_idx, ten_id, "<")
        and_op_idx = push_binary_op(arena, gt_op_idx, lt_op_idx, ".and.")

        ! Create assignment: b = c
        b_idx = push_identifier(arena, "b")
        c_idx = push_identifier(arena, "c")
        assign_idx = push_assignment(arena, b_idx, c_idx)

        ! Create where construct
        where_idx = push_where_construct(arena, and_op_idx, [assign_idx])
        prog_idx = push_program(arena, "test", [where_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper complex mask generation
            if ((index(output, "arith.andi") > 0 .or. &
                 index(output, "scf.if") > 0) .and. &
                (index(output, "arith.cmpi") > 0)) then
                print *, "PASS: Where with complex mask generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper complex mask generation"
                print *, "Expected: logical operations with comparisons"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_where_with_mask_expression

    ! Helper function to count occurrences of a substring
    function count_occurrences(string, substring) result(count)
        character(len=*), intent(in) :: string, substring
        integer :: count
        integer :: pos, start

        count = 0
        start = 1
        do
            pos = index(string(start:), substring)
            if (pos == 0) exit
            count = count + 1
            start = start + pos + len(substring) - 1
        end do
    end function count_occurrences

end program test_where_constructs
