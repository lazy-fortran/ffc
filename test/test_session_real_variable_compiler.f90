program test_session_real_variable_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== direct session real variable compiler test ==='

    if (.not. expect_output( &
         'program main'//new_line('a')// &
         '  real :: x'//new_line('a')// &
         '  real :: y'//new_line('a')// &
         '  x = 2.0'//new_line('a')// &
         '  y = x + 1.5'//new_line('a')// &
         '  print *, y'//new_line('a')// &
         'end program main', '3.500000'//new_line('a'), &
         '/tmp/ffc_session_real_var_test')) stop 1

    if (.not. expect_output( &
         'program main'//new_line('a')// &
         '  real :: x'//new_line('a')// &
         '  x = sign(2.5, -4.0)'//new_line('a')// &
         '  print *, x'//new_line('a')// &
         'end program main', '-2.500000'//new_line('a'), &
         '/tmp/ffc_session_real_sign_test')) stop 1

    print *, 'PASS: real variables and arithmetic lower through direct LIRIC'
end program test_session_real_variable_compiler
