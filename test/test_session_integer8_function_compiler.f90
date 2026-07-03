program test_session_integer8_function_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    print *, '=== direct session integer(8) function compiler test ==='

    ! A non-contained integer(8) module function returns through the i64 ABI.
    ! The result exceeds 2**31, so a correct i64 path is required: an i32
    ! truncation would leave a different value and trip the error stop.
    if (.not. expect_exit_status( &
        'module m'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        'contains'//new_line('a')// &
        '  integer(8) function scaled(a)'//new_line('a')// &
        '    integer(8), intent(in) :: a'//new_line('a')// &
        '    scaled = a * 2_8'//new_line('a')// &
        '  end function scaled'//new_line('a')// &
        'end module m'//new_line('a')// &
        'program main'//new_line('a')// &
        '  use m'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  integer(8) :: r'//new_line('a')// &
        '  r = scaled(2000000000_8)'//new_line('a')// &
        '  if (r /= 4000000000_8) error stop 3'//new_line('a')// &
        'end program main', 0, &
        '/tmp/ffc_session_integer8_fn_test')) stop 1

    print *, 'PASS: integer(8) module function lowers through the i64 ABI'
end program test_session_integer8_function_compiler
