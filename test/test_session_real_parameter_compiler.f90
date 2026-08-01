program test_session_real_parameter_compiler
    use ffc_test_support, only: expect_error_contains, expect_output
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

    ! Old-style PARAMETER statement: the constant takes its implicit type
    ! (i-n is integer) and folds into a later array bound.
    if (.not. expect_output( &
        'program main'//nl// &
        '  parameter (ialen = 42)'//nl// &
        '  integer :: myarray(ialen)'//nl// &
        '  myarray = 0'//nl// &
        '  print *, size(myarray)'//nl// &
        '  print *, ialen'//nl// &
        'end program main', &
        '          42'//nl// &
        '          42'//nl, &
        '/tmp/ffc_session_old_parameter_test')) stop 1

    ! A PARAMETER initializer that is not a constant expression is rejected.
    if (.not. expect_error_contains( &
        'program main'//nl// &
        '  integer :: k'//nl// &
        '  parameter (ialen = k)'//nl// &
        '  integer :: myarray(ialen)'//nl// &
        '  print *, size(myarray)'//nl// &
        'end program main', &
        'compile-time integer parameter was not declared', &
        '/tmp/ffc_session_old_parameter_nonconst_test')) stop 1

    print *, 'PASS: real/logical/character parameters lower through LIRIC'
end program test_session_real_parameter_compiler
