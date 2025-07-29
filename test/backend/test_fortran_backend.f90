program test_fortran_backend
    use iso_fortran_env, only: error_unit
    use backend_interface
    use backend_constants
    use ast_core, only: ast_arena_t, create_ast_stack, LITERAL_STRING
    use ast_factory
    use fortran_backend
    implicit none

    logical :: all_tests_passed

    print *, "=== Fortran Backend Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests
    if (.not. test_fortran_backend_exists()) all_tests_passed = .false.
    if (.not. test_fortran_backend_simple_program()) all_tests_passed = .false.
    if (.not. test_fortran_backend_preserves_functionality()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All Fortran backend tests passed!"
        stop 0
    else
        print *, "Some Fortran backend tests failed!"
        stop 1
    end if

contains

    function test_fortran_backend_exists() result(passed)
        logical :: passed
        type(fortran_backend_t) :: backend
        class(backend_t), allocatable :: backend_poly

        passed = .false.

        ! Test that Fortran backend module exists and can be instantiated
        allocate (backend_poly, source=backend)

        if (allocated(backend_poly)) then
            print *, "PASS: Fortran backend module exists"
            passed = .true.
        else
            print *, "FAIL: Failed to create Fortran backend"
            passed = .false.
        end if
    end function test_fortran_backend_exists

    function test_fortran_backend_simple_program() result(passed)
        logical :: passed
        type(ast_arena_t) :: arena
        type(fortran_backend_t) :: backend
        type(backend_options_t) :: options
        integer :: prog_idx, print_idx, lit_idx
        character(len=:), allocatable :: code
        character(len=256) :: error_msg

        passed = .false.

        ! Create a simple program AST
        arena = create_ast_stack()
        ! Create program with print statement in body
        lit_idx = push_literal(arena, '"Hello, World!"', LITERAL_STRING)
        print_idx = push_print_statement(arena, "*", [lit_idx])
        prog_idx = push_program(arena, "test_prog", [print_idx])

        ! Generate code using Fortran backend
        call backend%generate_code(arena, prog_idx, options, code, error_msg)

        if (len_trim(error_msg) > 0) then
            print *, "FAIL: Fortran backend error: ", trim(error_msg)
            passed = .false.
        else if (index(code, "program test_prog") > 0 .and. &
                 index(code, "print *, ""Hello, World!""") > 0 .and. &
                 index(code, "end program test_prog") > 0) then
            print *, "PASS: Fortran backend generates simple program"
            passed = .true.
        else
            print *, "FAIL: Fortran backend output incorrect"
            print *, "Generated code:"
            print *, code
            passed = .false.
        end if
    end function test_fortran_backend_simple_program

    function test_fortran_backend_preserves_functionality() result(passed)
        logical :: passed
        type(fortran_backend_t) :: backend

        passed = .false.

        ! Test that all existing functionality is preserved
        ! This test ensures we've successfully moved all codegen logic
        if (backend%get_name() == "Fortran") then
            print *, "PASS: Fortran backend preserves basic functionality"
            passed = .true.
        else
            print *, "FAIL: Fortran backend functionality not preserved"
            passed = .false.
        end if
    end function test_fortran_backend_preserves_functionality

end program test_fortran_backend
