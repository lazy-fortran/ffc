program test_session_integer_subroutine_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== direct session integer subroutine compiler test ==='

    if (.not. expect_output( &
         'program main'//new_line('a')// &
         '  integer :: x'//new_line('a')// &
         '  x = 5'//new_line('a')// &
         '  call bump(x)'//new_line('a')// &
         '  print *, x'//new_line('a')// &
         'contains'//new_line('a')// &
         '  subroutine bump(x)'//new_line('a')// &
         '    integer, intent(inout) :: x'//new_line('a')// &
         '    x = x + 2'//new_line('a')// &
         '  end subroutine bump'//new_line('a')// &
         'end program main', '           7'//new_line('a'), &
         '/tmp/ffc_session_integer_sub_test')) stop 1

    print *, 'PASS: integer subroutine reference args lower through direct LIRIC session'
end program test_session_integer_subroutine_compiler
