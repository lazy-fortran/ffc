program test_session_logical_function_print_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session logical function print compiler test ==='

    all_passed = .true.
    if (.not. test_module_logical_function_print()) all_passed = .false.
    if (.not. test_contained_logical_function_print()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: logical function calls print through direct LIRIC session'

contains

    logical function test_module_logical_function_print()
        ! A logical-valued module function printed directly formats as T/F
        ! rather than falling through to the integer print path (#290 W3).
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  logical function ispos(x)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    ispos = x > 0'//new_line('a')// &
            '  end function'//new_line('a')// &
            'end module'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  print *, ispos(5), ispos(-2)'//new_line('a')// &
            'end program main'

        test_module_logical_function_print = expect_output( &
            source, ' T F'//new_line('a'), &
            '/tmp/ffc_session_logical_module_function_print_test')
    end function test_module_logical_function_print

    logical function test_contained_logical_function_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  print *, iseven(4)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  logical function iseven(n)'//new_line('a')// &
            '    integer, intent(in) :: n'//new_line('a')// &
            '    iseven = mod(n, 2) == 0'//new_line('a')// &
            '  end function'//new_line('a')// &
            'end program main'

        test_contained_logical_function_print = expect_output( &
            source, ' T'//new_line('a'), &
            '/tmp/ffc_session_logical_contained_function_print_test')
    end function test_contained_logical_function_print

end program test_session_logical_function_print_compiler
