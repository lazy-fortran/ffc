program test_do_while_loops
    use iso_fortran_env, only: error_unit
    use mlir_backend
    use backend_factory
    use backend_interface
    use fortfront, only: ast_arena_t, create_ast_arena, LITERAL_INTEGER
    use ast_factory
    use mlir_utils, only: int_to_str
    implicit none

    logical :: all_tests_passed

    print *, "=== Do-While Loop Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all do-while loop tests
    if (.not. test_simple_do_while()) all_tests_passed = .false.
    if (.not. test_do_while_with_body()) all_tests_passed = .false.
    if (.not. test_do_while_with_counter()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All do-while loop tests passed!"
        stop 0
    else
        print *, "Some do-while loop tests failed!"
        stop 1
    end if

contains

    function test_simple_do_while() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: cond_idx, while_idx, prog_idx, i_id, ten_id, less_op_idx

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
        !       end do
        arena = create_ast_arena()

        ! Create condition: i < 10
        i_id = push_identifier(arena, "i")
        ten_id = push_literal(arena, "10", LITERAL_INTEGER)
        less_op_idx = push_binary_op(arena, i_id, ten_id, "<")

        ! Create do-while loop
        while_idx = push_do_while(arena, less_op_idx, [integer ::])
        prog_idx = push_program(arena, "test", [while_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper do-while loop generation
            if (index(output, "scf.while") > 0 .or. &
                index(output, "cf.br") > 0 .and. &
                index(output, "cf.cond_br") > 0) then
                print *, "PASS: Simple do-while generates proper MLIR loop"
                passed = .true.
            else
                print *, "FAIL: Missing proper do-while loop generation"
                print *, "Expected: scf.while or cf.br/cf.cond_br"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_simple_do_while

    function test_do_while_with_body() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: cond_idx, while_idx, prog_idx, i_id, five_id, gt_op_idx
        integer :: assign_idx, one_id, add_idx

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: do while (i > 5)
        !         i = i + 1
        !       end do
        arena = create_ast_arena()

        ! Create condition: i > 5
        i_id = push_identifier(arena, "i")
        five_id = push_literal(arena, "5", LITERAL_INTEGER)
        gt_op_idx = push_binary_op(arena, i_id, five_id, ">")

        ! Create body: i = i + 1
        one_id = push_literal(arena, "1", LITERAL_INTEGER)
        add_idx = push_binary_op(arena, i_id, one_id, "+")
        assign_idx = push_assignment(arena, i_id, add_idx)

        ! Create do-while loop with body
        while_idx = push_do_while(arena, gt_op_idx, [assign_idx])
        prog_idx = push_program(arena, "test", [while_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for proper do-while loop with body
            if ((index(output, "scf.while") > 0 .or. &
                 index(output, "cf.br") > 0 .and. &
                 index(output, "cf.cond_br") > 0) .and. &
                index(output, "arith.addi") > 0) then
                print *, "PASS: Do-while with body generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing proper do-while body generation"
                print *, "Expected: loop structure with arith.addi"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_do_while_with_body

    function test_do_while_with_counter() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        class(backend_t), allocatable :: backend
        character(len=256) :: factory_error
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg
        integer :: decl_idx, cond_idx, while_idx, prog_idx
        integer :: i_id, zero_id, hundred_id, lt_op_idx, assign_idx, add_idx, one_id

        passed = .false.
        factory_error = ""

        ! Create MLIR backend
        call create_backend("mlir", backend, factory_error)
        if (len_trim(factory_error) > 0) then
            print *, "FAIL: Could not create MLIR backend: ", trim(factory_error)
            return
        end if

        error_msg = ""

        ! Test: integer :: i = 0
        !       do while (i < 100)
        !         i = i + 1
        !       end do
        arena = create_ast_arena()

        ! Create declaration: integer :: i = 0
        zero_id = push_literal(arena, "0", LITERAL_INTEGER)
        decl_idx = push_declaration(arena, "integer", ["i"], initializer_index=zero_id)

        ! Create condition: i < 100
        i_id = push_identifier(arena, "i")
        hundred_id = push_literal(arena, "100", LITERAL_INTEGER)
        lt_op_idx = push_binary_op(arena, i_id, hundred_id, "<")

        ! Create body: i = i + 1
        one_id = push_literal(arena, "1", LITERAL_INTEGER)
        add_idx = push_binary_op(arena, i_id, one_id, "+")
        assign_idx = push_assignment(arena, i_id, add_idx)

        ! Create do-while loop
        while_idx = push_do_while(arena, lt_op_idx, [assign_idx])
        prog_idx = push_program(arena, "test", [decl_idx, while_idx])

        ! Generate MLIR code
        options%compile_mode = .true.
        call backend%generate_code(arena, prog_idx, options, output, error_msg)

        if (len_trim(error_msg) == 0 .and. allocated(output)) then
            ! Check for complete do-while implementation
            if ((index(output, "scf.while") > 0 .or. &
                 index(output, "cf.br") > 0 .and. &
                 index(output, "cf.cond_br") > 0) .and. &
                index(output, "memref.alloca") > 0 .and. &
                index(output, "arith.cmpi") > 0) then
                print *, "PASS: Complete do-while with counter generates proper MLIR"
                passed = .true.
            else
                print *, "FAIL: Missing complete do-while implementation"
                print *, "Expected: loop, variable allocation, comparison"
                print *, "Output:"
                print *, output
            end if
        else
            print *, "FAIL: Error generating MLIR: ", trim(error_msg)
        end if
    end function test_do_while_with_counter

end program test_do_while_loops
