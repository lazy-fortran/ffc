program test_session_compound_declarations_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== compound declarations compiler test ==='

    all_passed = .true.
    if (.not. test_two_integers()) all_passed = .false.
    if (.not. test_three_integers()) all_passed = .false.
    if (.not. test_logicals()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: compound declarations lower through direct LIRIC session'

contains

    logical function test_two_integers()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  i = 7'//new_line('a')// &
            '  j = 5'//new_line('a')// &
            '  stop i + j'//new_line('a')// &
            'end program main'

        test_two_integers = expect_exit_status( &
            source, 12, '/tmp/ffc_compound_two_int_test')
    end function test_two_integers

    logical function test_three_integers()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a, b, c'//new_line('a')// &
            '  a = 10'//new_line('a')// &
            '  b = 3'//new_line('a')// &
            '  c = 4'//new_line('a')// &
            '  stop a + b + c'//new_line('a')// &
            'end program main'

        test_three_integers = expect_exit_status( &
            source, 17, '/tmp/ffc_compound_three_int_test')
    end function test_three_integers

    logical function test_logicals()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  logical :: x, y'//new_line('a')// &
            '  x = .true.'//new_line('a')// &
            '  y = .false.'//new_line('a')// &
            '  if (x) then'//new_line('a')// &
            '    stop 1'//new_line('a')// &
            '  else'//new_line('a')// &
            '    stop 2'//new_line('a')// &
            '  end if'//new_line('a')// &
            'end program main'

        test_logicals = expect_exit_status( &
            source, 1, '/tmp/ffc_compound_logicals_test')
    end function test_logicals

end program test_session_compound_declarations_compiler
