program test_session_real_kind_expr_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== direct session real-kind expression compiler test ==='

    ! Mixed int/real arithmetic: the integer operand coerces to f64.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  use, intrinsic :: iso_fortran_env, only: dp => real64'// &
        new_line('a')// &
        '  integer :: i'//new_line('a')// &
        '  real(dp) :: y'//new_line('a')// &
        '  i = 7'//new_line('a')// &
        '  y = i / 2.0_dp'//new_line('a')// &
        '  print *, y'//new_line('a')// &
        'end program main', '   3.5000000000000000     '//new_line('a'), &
        '/tmp/ffc_real_kind_mixed_f64')) stop 1

    ! real(x, kind=dp) two-argument conversion yields f64.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  use, intrinsic :: iso_fortran_env, only: dp => real64'// &
        new_line('a')// &
        '  integer :: i'//new_line('a')// &
        '  real(dp) :: y'//new_line('a')// &
        '  i = 7'//new_line('a')// &
        '  y = real(i, kind=dp)'//new_line('a')// &
        '  print *, y'//new_line('a')// &
        'end program main', '   7.0000000000000000     '//new_line('a'), &
        '/tmp/ffc_real_kind_2arg_kw')) stop 1

    ! real(x, 8) two-argument conversion yields f64.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  use, intrinsic :: iso_fortran_env, only: dp => real64'// &
        new_line('a')// &
        '  integer :: i'//new_line('a')// &
        '  real(dp) :: y'//new_line('a')// &
        '  i = 7'//new_line('a')// &
        '  y = real(i, 8)'//new_line('a')// &
        '  print *, y'//new_line('a')// &
        'end program main', '   7.0000000000000000     '//new_line('a'), &
        '/tmp/ffc_real_kind_2arg_pos')) stop 1

    ! real(dp) scalar prints with f64 formatting matching gfortran.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  use, intrinsic :: iso_fortran_env, only: dp => real64'// &
        new_line('a')// &
        '  integer :: i'//new_line('a')// &
        '  real(dp) :: y'//new_line('a')// &
        '  i = 7'//new_line('a')// &
        '  y = i * 1.0_dp + 3'//new_line('a')// &
        '  print *, y'//new_line('a')// &
        'end program main', '   10.000000000000000     '//new_line('a'), &
        '/tmp/ffc_real_kind_mixed_add')) stop 1

    ! Mixed int/real arithmetic in default real (f32) coerces correctly.
    if (.not. expect_output( &
        'program main'//new_line('a')// &
        '  integer :: i'//new_line('a')// &
        '  real :: y'//new_line('a')// &
        '  i = 7'//new_line('a')// &
        '  y = i / 2.0'//new_line('a')// &
        '  print *, y'//new_line('a')// &
        'end program main', '   3.50000000    '//new_line('a'), &
        '/tmp/ffc_real_kind_mixed_f32')) stop 1

    print *, 'PASS: mixed int/real arithmetic and real(x,kind) lower correctly'
end program test_session_real_kind_expr_compiler
