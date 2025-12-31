module test_backend_interface_impl
    use backend_interface
    use fortfront
    implicit none
    private

    public :: test_backend_t, test_backend_error_t

    ! Test backend implementation
    type, extends(backend_t) :: test_backend_t
    contains
        procedure :: generate_code => test_backend_generate_code
        procedure :: get_name => test_backend_get_name
        procedure :: get_version => test_backend_get_version
    end type test_backend_t

    ! Test backend that returns an error
    type, extends(backend_t) :: test_backend_error_t
    contains
        procedure :: generate_code => test_backend_error_generate_code
        procedure :: get_name => test_backend_error_get_name
        procedure :: get_version => test_backend_error_get_version
    end type test_backend_error_t

contains

    subroutine test_backend_generate_code(this, arena, prog_index, options, &
                                          output, error_msg)
        class(test_backend_t), intent(inout) :: this
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: prog_index
        type(backend_options_t), intent(in) :: options
        character(len=:), allocatable, intent(out) :: output
        character(len=*), intent(out) :: error_msg

        output = "test output"
        error_msg = ""
    end subroutine test_backend_generate_code

    function test_backend_get_name(this) result(name)
        class(test_backend_t), intent(in) :: this
        character(len=:), allocatable :: name
        name = "test"
    end function test_backend_get_name

    function test_backend_get_version(this) result(version)
        class(test_backend_t), intent(in) :: this
        character(len=:), allocatable :: version
        version = "1.0.0"
    end function test_backend_get_version

    subroutine test_backend_error_generate_code(this, arena, prog_index, &
                                                options, output, error_msg)
        class(test_backend_error_t), intent(inout) :: this
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: prog_index
        type(backend_options_t), intent(in) :: options
        character(len=:), allocatable, intent(out) :: output
        character(len=*), intent(out) :: error_msg

        output = ""
        error_msg = "Test error message"
    end subroutine test_backend_error_generate_code

    function test_backend_error_get_name(this) result(name)
        class(test_backend_error_t), intent(in) :: this
        character(len=:), allocatable :: name
        name = "test_error"
    end function test_backend_error_get_name

    function test_backend_error_get_version(this) result(version)
        class(test_backend_error_t), intent(in) :: this
        character(len=:), allocatable :: version
        version = "1.0.0"
    end function test_backend_error_get_version

end module test_backend_interface_impl

program test_backend_interface
    use iso_fortran_env, only: error_unit
    use backend_interface
    use backend_constants
    use fortfront
    use test_backend_interface_impl
    implicit none

    logical :: all_tests_passed

    print *, "=== Backend Interface Tests ==="
    print *

    all_tests_passed = .true.

    ! Run all tests
    if (.not. test_backend_interface_compiles()) all_tests_passed = .false.
    if (.not. test_backend_polymorphic_dispatch()) all_tests_passed = .false.
    if (.not. test_backend_error_handling()) all_tests_passed = .false.

    print *
    if (all_tests_passed) then
        print *, "All backend interface tests passed!"
        stop 0
    else
        print *, "Some backend interface tests failed!"
        stop 1
    end if

contains

    function test_backend_interface_compiles() result(passed)
        logical :: passed
        class(backend_t), allocatable :: backend

        passed = .false.

        ! Test that backend interface compiles
        ! The existence of the backend_t type is the test
        print *, "PASS: Backend interface compiles"
        passed = .true.
    end function test_backend_interface_compiles

    function test_backend_polymorphic_dispatch() result(passed)
        logical :: passed
        type(test_backend_t) :: test_backend
        class(backend_t), allocatable :: backend
        type(ast_arena_t) :: arena
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg

        passed = .false.

        ! Create a test backend instance
        allocate (backend, source=test_backend)

        ! Test polymorphic dispatch
        arena = create_ast_arena()
        call backend%generate_code(arena, 1, options, output, error_msg)

        if (output == "test output") then
            print *, "PASS: Backend polymorphic dispatch"
            passed = .true.
        else
            print *, "FAIL: Backend polymorphic dispatch failed"
        end if
    end function test_backend_polymorphic_dispatch

    function test_backend_error_handling() result(passed)
        logical :: passed
        type(test_backend_error_t) :: test_backend
        class(backend_t), allocatable :: backend
        type(ast_arena_t) :: arena
        type(backend_options_t) :: options
        character(len=:), allocatable :: output
        character(len=256) :: error_msg

        passed = .false.

        ! Create a test backend that returns an error
        allocate (backend, source=test_backend)

        arena = create_ast_arena()
        call backend%generate_code(arena, 1, options, output, error_msg)

        if (len_trim(error_msg) > 0) then
            print *, "PASS: Backend error handling"
            passed = .true.
        else
            print *, "FAIL: Backend should handle errors"
        end if
    end function test_backend_error_handling

end program test_backend_interface
