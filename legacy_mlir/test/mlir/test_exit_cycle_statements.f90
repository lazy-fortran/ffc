program test_exit_cycle_statements
    use iso_fortran_env, only: error_unit
    use mlir_backend
    use backend_factory
    use backend_interface
    use ast_core, only: ast_arena_t, create_ast_arena, LITERAL_INTEGER
    use ast_factory
    use mlir_utils, only: int_to_str
    implicit none

    logical :: all_tests_passed

    print *, "=== Exit and Cycle Statement Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all exit and cycle tests
    if (.not. test_exit_statement()) all_tests_passed = .false.
    if (.not. test_cycle_statement()) all_tests_passed = .false.
    if (.not. test_exit_in_do_loop()) all_tests_passed = .false.
    if (.not. test_cycle_in_do_while()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All exit and cycle statement tests passed!"
        stop 0
    else
        print *, "Some exit and cycle statement tests failed!"
        stop 1
    end if

contains

    function test_exit_statement() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: exit_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: exit
        arena = create_ast_arena()

        ! Create exit statement
        exit_idx = push_exit(arena)
        prog_idx = push_program(arena, "test", [exit_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper exit statement generation
            if (index(output, "cf.br") > 0 .or. &
                index(output, "// Exit statement") > 0) then
                print *, "PASS: Exit statement generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper exit statement generation"
                print *, "Expected: cf.br or exit comment"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_exit_statement

    function test_cycle_statement() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: cycle_idx, prog_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: cycle
        arena = create_ast_arena()

        ! Create cycle statement
        cycle_idx = push_cycle(arena)
        prog_idx = push_program(arena, "test", [cycle_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper cycle statement generation
            if (index(output, "cf.br") > 0 .or. &
                index(output, "// Cycle statement") > 0) then
                print *, "PASS: Cycle statement generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper cycle statement generation"
                print *, "Expected: cf.br or cycle comment"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_cycle_statement

    function test_exit_in_do_loop() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: start_idx, end_idx, step_idx, exit_idx, loop_idx, prog_idx
        integer :: i_id, one_id, ten_id

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: do i = 1, 10
        !         exit
        !       end do
        arena = create_ast_arena()

        ! Create loop bounds
        one_id = push_literal(arena, "1", LITERAL_INTEGER)
        ten_id = push_literal(arena, "10", LITERAL_INTEGER)

        ! Create exit statement
        exit_idx = push_exit(arena)

        ! Create do loop with exit in body
        loop_idx = push_do_loop(arena, "i", one_id, ten_id, body_indices=[exit_idx])
        prog_idx = push_program(arena, "test", [loop_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper exit in loop
            if ((index(output, "scf.for") > 0 .or. &
                 index(output, "cf.br") > 0) .and. &
                (index(output, "cf.br") > 0 .or. &
                 index(output, "// Exit statement") > 0)) then
                print *, "PASS: Exit in do loop generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper exit in do loop"
                print *, "Expected: loop structure with exit"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_exit_in_do_loop

    function test_cycle_in_do_while() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: cond_idx, cycle_idx, while_idx, prog_idx
        integer :: i_id, ten_id, lt_op_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: do while (i < 10)
        !         cycle
        !       end do
        arena = create_ast_arena()

        ! Create condition: i < 10
        i_id = push_identifier(arena, "i")
        ten_id = push_literal(arena, "10", LITERAL_INTEGER)
        lt_op_idx = push_binary_op(arena, i_id, ten_id, "<")

        ! Create cycle statement
        cycle_idx = push_cycle(arena)

        ! Create do-while loop with cycle in body
        while_idx = push_do_while(arena, lt_op_idx, [cycle_idx])
        prog_idx = push_program(arena, "test", [while_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper cycle in do-while
            if ((index(output, "cf.br ^loop_start") > 0 .or. &
                 index(output, "cf.cond_br") > 0) .and. &
                (index(output, "cf.br") > 0 .or. &
                 index(output, "// Cycle statement") > 0)) then
                print *, "PASS: Cycle in do-while generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper cycle in do-while"
                print *, "Expected: do-while structure with cycle"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_cycle_in_do_while

end program test_exit_cycle_statements
