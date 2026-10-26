program test_session_standalone_unit_compiler
    use ffc_test_support, only: expect_object_exists
    implicit none

    logical :: all_passed

    print *, '=== direct session standalone subroutine/function compiler test ==='

    all_passed = .true.
    if (.not. test_standalone_subroutine()) all_passed = .false.
    if (.not. test_standalone_integer_function()) all_passed = .false.
    if (.not. test_standalone_subroutine_with_arg()) all_passed = .false.
    if (.not. test_standalone_function_with_arg()) all_passed = .false.
    if (.not. test_standalone_subroutine_two_args()) all_passed = .false.
    if (.not. test_standalone_function_two_args()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: standalone subroutine and function units lower to objects'

contains

    logical function test_standalone_subroutine()
        ! A bare subroutine (no program wrapper) must lower to an object.
        character(len=*), parameter :: source = &
            'subroutine greet()'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'end subroutine greet'

        test_standalone_subroutine = expect_object_exists( &
            source, '/tmp/ffc_standalone_sub_test.o')
    end function test_standalone_subroutine

    logical function test_standalone_integer_function()
        ! A bare integer function (no program wrapper) must lower to an object.
        character(len=*), parameter :: source = &
            'integer function answer()'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  answer = 42'//new_line('a')// &
            'end function answer'

        test_standalone_integer_function = expect_object_exists( &
            source, '/tmp/ffc_standalone_func_test.o')
    end function test_standalone_integer_function

    logical function test_standalone_subroutine_with_arg()
        ! Standalone subroutine with an intent(inout) integer argument.
        character(len=*), parameter :: source = &
            'subroutine increment(x)'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, intent(inout) :: x'//new_line('a')// &
            '  x = x + 1'//new_line('a')// &
            'end subroutine increment'

        test_standalone_subroutine_with_arg = expect_object_exists( &
            source, '/tmp/ffc_standalone_sub_arg_test.o')
    end function test_standalone_subroutine_with_arg

    logical function test_standalone_function_with_arg()
        ! Standalone integer function with one intent(in) argument.
        character(len=*), parameter :: source = &
            'integer function double(n)'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, intent(in) :: n'//new_line('a')// &
            '  double = n * 2'//new_line('a')// &
            'end function double'

        test_standalone_function_with_arg = expect_object_exists( &
            source, '/tmp/ffc_standalone_func_arg_test.o')
    end function test_standalone_function_with_arg

    logical function test_standalone_subroutine_two_args()
        ! Standalone subroutine with two integer arguments.
        character(len=*), parameter :: source = &
            'subroutine add_to(a, b)'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, intent(inout) :: a'//new_line('a')// &
            '  integer, intent(in) :: b'//new_line('a')// &
            '  a = a + b'//new_line('a')// &
            'end subroutine add_to'

        test_standalone_subroutine_two_args = expect_object_exists( &
            source, '/tmp/ffc_standalone_sub2_test.o')
    end function test_standalone_subroutine_two_args

    logical function test_standalone_function_two_args()
        ! Standalone integer function with two integer arguments.
        character(len=*), parameter :: source = &
            'integer function add(a, b)'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, intent(in) :: a'//new_line('a')// &
            '  integer, intent(in) :: b'//new_line('a')// &
            '  add = a + b'//new_line('a')// &
            'end function add'

        test_standalone_function_two_args = expect_object_exists( &
            source, '/tmp/ffc_standalone_func2_test.o')
    end function test_standalone_function_two_args

end program test_session_standalone_unit_compiler
