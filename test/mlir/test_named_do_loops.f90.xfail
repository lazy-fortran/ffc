program test_named_do_loops
    use iso_fortran_env, only: error_unit
    use mlir_backend
    use backend_factory
    use backend_interface
    use fortfront, only: ast_arena_t, create_ast_arena, LITERAL_INTEGER
    use ast_factory
    use mlir_utils, only: int_to_str
    implicit none

    logical :: all_tests_passed

    print *, "=== Named Do Loop Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all named do loop tests
    if (.not. test_simple_named_do()) all_tests_passed = .false.
    if (.not. test_nested_named_do()) all_tests_passed = .false.
    if (.not. test_exit_with_label()) all_tests_passed = .false.
    if (.not. test_cycle_with_label()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All named do loop tests passed!"
        stop 0
    else
        print *, "Some named do loop tests failed!"
        stop 1
    end if

contains

    function test_simple_named_do() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: loop_idx, prog_idx, one_id, ten_id

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: outer: do i = 1, 10
        !       end do outer
        arena = create_ast_arena()

        ! Create loop bounds
        one_id = push_literal(arena, "1", LITERAL_INTEGER)
        ten_id = push_literal(arena, "10", LITERAL_INTEGER)

        ! Create named do loop
        loop_idx = push_do_loop(arena, "i", one_id, ten_id, &
                                body_indices=[integer ::], loop_label="outer")
        prog_idx = push_program(arena, "test", [loop_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper named do loop generation
            if ((index(output, "scf.for") > 0 .or. &
                 index(output, "cf.br") > 0) .and. &
                (index(output, "outer") > 0 .or. &
                 index(output, "^outer_") > 0)) then
                print *, "PASS: Named do loop generates proper MLIR with labels"
                passed = .true.
            else
                print *, "FAIL: Missing proper named do loop generation"
                print *, "Expected: loop structure with label 'outer'"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_simple_named_do

    function test_nested_named_do() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: outer_idx, inner_idx, prog_idx
        integer :: one_id, ten_id, five_id

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: outer: do i = 1, 10
        !         inner: do j = 1, 5
        !         end do inner
        !       end do outer
        arena = create_ast_arena()

        ! Create loop bounds
        one_id = push_literal(arena, "1", LITERAL_INTEGER)
        ten_id = push_literal(arena, "10", LITERAL_INTEGER)
        five_id = push_literal(arena, "5", LITERAL_INTEGER)

        ! Create inner named do loop
        inner_idx = push_do_loop(arena, "j", one_id, five_id, &
                                 body_indices=[integer ::], loop_label="inner")

        ! Create outer named do loop with inner loop as body
        outer_idx = push_do_loop(arena, "i", one_id, ten_id, &
                                 body_indices=[inner_idx], loop_label="outer")
        prog_idx = push_program(arena, "test", [outer_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper nested named do loops
            if ((index(output, "scf.for") > 0 .or. &
                 index(output, "cf.br") > 0) .and. &
                (index(output, "outer") > 0 .and. &
                 index(output, "inner") > 0)) then
                print *, "PASS: Nested named do loops generate proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper nested named do loop generation"
                print *, "Expected: nested loops with 'outer' and 'inner' labels"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_nested_named_do

    function test_exit_with_label() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: outer_idx, inner_idx, exit_idx, prog_idx
        integer :: one_id, ten_id, five_id

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: outer: do i = 1, 10
        !         inner: do j = 1, 5
        !           exit outer
        !         end do inner
        !       end do outer
        arena = create_ast_arena()

        ! Create loop bounds
        one_id = push_literal(arena, "1", LITERAL_INTEGER)
        ten_id = push_literal(arena, "10", LITERAL_INTEGER)
        five_id = push_literal(arena, "5", LITERAL_INTEGER)

        ! Create exit statement with label
        exit_idx = push_exit(arena, loop_label="outer")

        ! Create inner named do loop with exit
        inner_idx = push_do_loop(arena, "j", one_id, five_id, &
                                 body_indices=[exit_idx], loop_label="inner")

        ! Create outer named do loop
        outer_idx = push_do_loop(arena, "i", one_id, ten_id, &
                                 body_indices=[inner_idx], loop_label="outer")
        prog_idx = push_program(arena, "test", [outer_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper exit with label
            if ((index(output, "cf.br") > 0) .and. &
                (index(output, "outer") > 0) .and. &
                (index(output, "^outer_end") > 0 .or. &
                 index(output, "exit outer") > 0)) then
                print *, "PASS: Exit with label generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper exit with label generation"
                print *, "Expected: cf.br to outer_end or exit comment"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_exit_with_label

    function test_cycle_with_label() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: outer_idx, inner_idx, cycle_idx, prog_idx
        integer :: one_id, ten_id, five_id

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: outer: do i = 1, 10
        !         inner: do j = 1, 5
        !           cycle outer
        !         end do inner
        !       end do outer
        arena = create_ast_arena()

        ! Create loop bounds
        one_id = push_literal(arena, "1", LITERAL_INTEGER)
        ten_id = push_literal(arena, "10", LITERAL_INTEGER)
        five_id = push_literal(arena, "5", LITERAL_INTEGER)

        ! Create cycle statement with label
        cycle_idx = push_cycle(arena, loop_label="outer")

        ! Create inner named do loop with cycle
        inner_idx = push_do_loop(arena, "j", one_id, five_id, &
                                 body_indices=[cycle_idx], loop_label="inner")

        ! Create outer named do loop
        outer_idx = push_do_loop(arena, "i", one_id, ten_id, &
                                 body_indices=[inner_idx], loop_label="outer")
        prog_idx = push_program(arena, "test", [outer_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper cycle with label
            if ((index(output, "cf.br") > 0) .and. &
                (index(output, "outer") > 0) .and. &
                (index(output, "^outer_start") > 0 .or. &
                 index(output, "cycle outer") > 0)) then
                print *, "PASS: Cycle with label generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper cycle with label generation"
                print *, "Expected: cf.br to outer_start or cycle comment"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_cycle_with_label

end program test_named_do_loops
