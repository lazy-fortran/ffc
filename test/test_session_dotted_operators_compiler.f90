program test_session_dotted_operators_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session dotted comparison operator test ==='

    all_passed = .true.
    if (.not. test_integer_dotted()) all_passed = .false.
    if (.not. test_integer_dotted_uppercase()) all_passed = .false.
    if (.not. test_real_dotted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: .eq./.ne./.lt./.gt./.le./.ge. lower like symbolic operators'

contains

    logical function test_integer_dotted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a = 3, b = 5'//new_line('a')// &
            '  if (a .ne. b .and. a .lt. b) stop 7'//new_line('a')// &
            '  stop 1'//new_line('a')// &
            'end program main'

        test_integer_dotted = expect_exit_status( &
            source, 7, '/tmp/ffc_session_dotted_int_test')
    end function test_integer_dotted

    logical function test_integer_dotted_uppercase()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a = 4'//new_line('a')// &
            '  if (a .EQ. 4 .and. a .GE. 4 .and. a .LE. 4) stop 9'//new_line('a')// &
            '  stop 1'//new_line('a')// &
            'end program main'

        test_integer_dotted_uppercase = expect_exit_status( &
            source, 9, '/tmp/ffc_session_dotted_upper_test')
    end function test_integer_dotted_uppercase

    logical function test_real_dotted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x = 1.5, y = 2.5'//new_line('a')// &
            '  if (x .lt. y .and. y .gt. x) stop 5'//new_line('a')// &
            '  stop 1'//new_line('a')// &
            'end program main'

        test_real_dotted = expect_exit_status( &
            source, 5, '/tmp/ffc_session_dotted_real_test')
    end function test_real_dotted
end program test_session_dotted_operators_compiler
