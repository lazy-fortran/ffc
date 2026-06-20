program test_session_real_parameter_compiler
    use ffc_test_support, only: expect_output
    implicit none
    character(len=1), parameter :: nl = new_line('a')

    print *, '=== direct session real/logical/character parameter test ==='

    if (.not. expect_output( &
         'program main'//nl// &
         '  real, parameter :: half = 0.5'//nl// &
         '  real(8), parameter :: two = 2.0d0'//nl// &
         '  logical, parameter :: yes = .true.'//nl// &
         '  character(len=*), parameter :: tag = "ok"'//nl// &
         '  real :: r'//nl// &
         '  r = half + 1.0'//nl// &
         '  print *, half'//nl// &
         '  print *, two'//nl// &
         '  print *, yes'//nl// &
         '  print *, tag'//nl// &
         '  print *, r'//nl// &
         '  if (yes) print *, "branch"'//nl// &
         'end program main', &
         '  0.500000000    '//nl// &
         '   2.0000000000000000     '//nl// &
         ' T'//nl// &
         ' ok'//nl// &
         '   1.50000000    '//nl// &
         ' branch'//nl, &
         '/tmp/ffc_session_real_parameter_test')) stop 1

    print *, 'PASS: real/logical/character parameters lower through LIRIC'
end program test_session_real_parameter_compiler
