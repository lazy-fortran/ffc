program test_select_case
    use iso_fortran_env, only: error_unit
    use mlir_backend
    use backend_factory
    use backend_interface
    use fortfront, only: ast_arena_t, create_ast_arena, LITERAL_INTEGER, LITERAL_STRING
    use ast_factory
    use mlir_utils, only: int_to_str
    implicit none

    logical :: all_tests_passed

    print *, "=== Select Case Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all select case tests
    if (.not. test_simple_integer_select()) all_tests_passed = .false.
    if (.not. test_character_select()) all_tests_passed = .false.
    if (.not. test_select_with_ranges()) all_tests_passed = .false.
    if (.not. test_select_with_default()) all_tests_passed = .false.
    if (.not. test_multiple_case_values()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All select case tests passed!"
        stop 0
    else
        print *, "Some select case tests failed!"
        stop 1
    end if

contains

    function test_simple_integer_select() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: select_idx, prog_idx, var_idx, case1_idx, case2_idx
        integer :: one_id, two_id, assign1_idx, assign2_idx, x_id, a_id

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: select case (i)
        !         case (1)
        !           x = a
        !         case (2)
        !           x = a
        !       end select
        arena = create_ast_arena()

        ! Create selector variable
        var_idx = push_identifier(arena, "i")

        ! Create case values
        one_id = push_literal(arena, "1", LITERAL_INTEGER)
        two_id = push_literal(arena, "2", LITERAL_INTEGER)

        ! Create assignments for each case
        x_id = push_identifier(arena, "x")
        a_id = push_identifier(arena, "a")
        assign1_idx = push_assignment(arena, x_id, a_id)
        assign2_idx = push_assignment(arena, x_id, a_id)

        ! Create case blocks
        case1_idx = push_case_block(arena, [one_id], [assign1_idx])
        case2_idx = push_case_block(arena, [two_id], [assign2_idx])

        ! Create select case construct
        select_idx = push_select_case(arena, var_idx, [case1_idx, case2_idx])
        prog_idx = push_program(arena, "test", [select_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper select case generation
            if ((index(output, "cf.switch") > 0 .or. &
                 index(output, "scf.if") > 0) .and. &
                (index(output, "case") > 0 .or. &
                 index(output, "select") > 0)) then
                print *, "PASS: Simple integer select case generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper select case generation"
                print *, "Expected: switch or conditional with case structure"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_simple_integer_select

    function test_character_select() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: select_idx, prog_idx, var_idx, case1_idx, case2_idx
        integer :: a_str_id, b_str_id, assign1_idx, assign2_idx, x_id, result_id

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: select case (char_var)
        !         case ('a')
        !           x = result
        !         case ('b')
        !           x = result
        !       end select
        arena = create_ast_arena()

        ! Create selector variable
        var_idx = push_identifier(arena, "char_var")

        ! Create case values
        a_str_id = push_literal(arena, "'a'", LITERAL_STRING)
        b_str_id = push_literal(arena, "'b'", LITERAL_STRING)

        ! Create assignments for each case
        x_id = push_identifier(arena, "x")
        result_id = push_identifier(arena, "result")
        assign1_idx = push_assignment(arena, x_id, result_id)
        assign2_idx = push_assignment(arena, x_id, result_id)

        ! Create case blocks
        case1_idx = push_case_block(arena, [a_str_id], [assign1_idx])
        case2_idx = push_case_block(arena, [b_str_id], [assign2_idx])

        ! Create select case construct
        select_idx = push_select_case(arena, var_idx, [case1_idx, case2_idx])
        prog_idx = push_program(arena, "test", [select_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper character select case
            if ((index(output, "cf.switch") > 0 .or. &
                 index(output, "scf.if") > 0) .and. &
                (index(output, "character") > 0 .or. &
                 index(output, "string") > 0 .or. &
                 index(output, "select") > 0)) then
                print *, "PASS: Character select case generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper character select case"
                print *, "Expected: character comparison with case structure"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_character_select

    function test_select_with_ranges() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: select_idx, prog_idx, var_idx, case1_idx, case2_idx
        integer :: range1_idx, range2_idx, assign1_idx, assign2_idx, x_id, val_id

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: select case (i)
        !         case (1:5)
        !           x = val
        !         case (10:20)
        !           x = val
        !       end select
        arena = create_ast_arena()

        ! Create selector variable
        var_idx = push_identifier(arena, "i")

        ! Create case ranges
        range1_idx = push_case_range(arena, 1, 5)
        range2_idx = push_case_range(arena, 10, 20)

        ! Create assignments for each case
        x_id = push_identifier(arena, "x")
        val_id = push_identifier(arena, "val")
        assign1_idx = push_assignment(arena, x_id, val_id)
        assign2_idx = push_assignment(arena, x_id, val_id)

        ! Create case blocks
        case1_idx = push_case_block(arena, [range1_idx], [assign1_idx])
        case2_idx = push_case_block(arena, [range2_idx], [assign2_idx])

        ! Create select case construct
        select_idx = push_select_case(arena, var_idx, [case1_idx, case2_idx])
        prog_idx = push_program(arena, "test", [select_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper range case generation
            if ((index(output, "arith.cmpi") > 0 .or. &
                 index(output, "scf.if") > 0) .and. &
                (index(output, "range") > 0 .or. &
                 index(output, ">=") > 0 .or. &
                 index(output, "<=") > 0)) then
                print *, "PASS: Select case with ranges generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper range case generation"
                print *, "Expected: range comparisons with conditionals"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_select_with_ranges

    function test_select_with_default() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: select_idx, prog_idx, var_idx, case1_idx, default_idx
       integer :: one_id, assign1_idx, assign_default_idx, x_id, val1_id, default_val_id

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: select case (i)
        !         case (1)
        !           x = val1
        !         case default
        !           x = default_val
        !       end select
        arena = create_ast_arena()

        ! Create selector variable
        var_idx = push_identifier(arena, "i")

        ! Create case value
        one_id = push_literal(arena, "1", LITERAL_INTEGER)

        ! Create assignments
        x_id = push_identifier(arena, "x")
        val1_id = push_identifier(arena, "val1")
        default_val_id = push_identifier(arena, "default_val")
        assign1_idx = push_assignment(arena, x_id, val1_id)
        assign_default_idx = push_assignment(arena, x_id, default_val_id)

        ! Create case blocks
        case1_idx = push_case_block(arena, [one_id], [assign1_idx])
        default_idx = push_case_default(arena, [assign_default_idx])

        ! Create select case construct with default
    select_idx = push_select_case_with_default(arena, var_idx, [case1_idx], default_idx)
        prog_idx = push_program(arena, "test", [select_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper default case generation
            if ((index(output, "cf.switch") > 0 .or. &
                 index(output, "scf.if") > 0) .and. &
                (index(output, "default") > 0 .or. &
                 index(output, "else") > 0)) then
                print *, "PASS: Select case with default generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper default case generation"
                print *, "Expected: switch with default or conditional with else"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_select_with_default

    function test_multiple_case_values() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: select_idx, prog_idx, var_idx, case1_idx
        integer :: one_id, three_id, five_id, assign_idx, x_id, val_id

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: select case (i)
        !         case (1, 3, 5)
        !           x = val
        !       end select
        arena = create_ast_arena()

        ! Create selector variable
        var_idx = push_identifier(arena, "i")

        ! Create multiple case values
        one_id = push_literal(arena, "1", LITERAL_INTEGER)
        three_id = push_literal(arena, "3", LITERAL_INTEGER)
        five_id = push_literal(arena, "5", LITERAL_INTEGER)

        ! Create assignment
        x_id = push_identifier(arena, "x")
        val_id = push_identifier(arena, "val")
        assign_idx = push_assignment(arena, x_id, val_id)

        ! Create case block with multiple values
        case1_idx = push_case_block(arena, [one_id, three_id, five_id], [assign_idx])

        ! Create select case construct
        select_idx = push_select_case(arena, var_idx, [case1_idx])
        prog_idx = push_program(arena, "test", [select_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper multiple value case generation
            if ((index(output, "arith.cmpi") > 0 .or. &
                 index(output, "scf.if") > 0) .and. &
                (index(output, "||") > 0 .or. &
                 index(output, "or") > 0 .or. &
                 index(output, "multiple") > 0)) then
                print *, "PASS: Multiple case values generate proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper multiple case value generation"
                print *, "Expected: multiple comparisons with OR logic"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_multiple_case_values

end program test_select_case
