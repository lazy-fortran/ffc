program test_session_size_i64_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    print *, '=== direct session SIZE integer(8) compiler test ==='

    if (.not. expect_exit_status( &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  integer :: a(3), b(2,4), n'//new_line('a')// &
        '  integer(8) :: size_a, size_b'//new_line('a')// &
        '  size_a = size(a, kind=8)'//new_line('a')// &
        '  size_b = size(b, dim=2, kind=8)'//new_line('a')// &
        '  if (size_a /= 3_8) error stop 1'//new_line('a')// &
        '  if (size_b /= 4_8) error stop 2'//new_line('a')// &
        '  n = 5'//new_line('a')// &
        '  call check_runtime_size(n)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine check_runtime_size(n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    integer :: c(n)'//new_line('a')// &
        '    integer(8) :: runtime_a, runtime_b'//new_line('a')// &
        '    runtime_a = size(c, kind=8)'//new_line('a')// &
        '    runtime_b = size(c, dim=1, kind=8)'//new_line('a')// &
        '    if (runtime_a /= 5_8) error stop 3'//new_line('a')// &
        '    if (runtime_b /= 5_8) error stop 4'//new_line('a')// &
        '  end subroutine check_runtime_size'//new_line('a')// &
        'end program main', 0, '/tmp/ffc_session_size_i64')) stop 1

    print *, 'PASS: SIZE widens fixed and runtime extents to integer(8)'
end program test_session_size_i64_compiler
