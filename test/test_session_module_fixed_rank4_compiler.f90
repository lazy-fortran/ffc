program test_session_module_fixed_rank4_compiler
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none

    character(len=*), parameter :: source = &
        'module state'//new_line('a')// &
        '  integer :: i4(2,2,2,2) = 2'//new_line('a')// &
        '  real :: r4(2,2,2,2) = 1.5'//new_line('a')// &
        '  logical :: flags(2,2,2,2) = .false.'//new_line('a')// &
        '  complex :: c4(2,2,2,2) = (1.0, -2.0)'//new_line('a')// &
        '  complex(8) :: c8(2,2,2,2) = (3.0d0, 4.0d0)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine mutate()'//new_line('a')// &
        '    i4(2,1,2,1) = 17'//new_line('a')// &
        '    r4(1,2,1,2) = 9.5'//new_line('a')// &
        '    flags(2,2,2,2) = .true.'//new_line('a')// &
        '    c4(1,1,1,1) = (5.0, -6.0)'//new_line('a')// &
        '    c8(2,2,2,2) = (-7.0d0, 8.0d0)'//new_line('a')// &
        '  end subroutine mutate'//new_line('a')// &
        'end module state'//new_line('a')// &
        'program main'//new_line('a')// &
        '  use state'//new_line('a')// &
        '  call mutate()'//new_line('a')// &
        '  if (sum(i4) /= 47) error stop 1'//new_line('a')// &
        '  if (sum(r4) /= 32.0) error stop 2'//new_line('a')// &
        '  if (count(flags) /= 1) error stop 3'//new_line('a')// &
        '  if (real(c4(1,1,1,1)) /= 5.0) error stop 4'//new_line('a')// &
        '  if (aimag(c4(1,1,1,1)) /= -6.0) error stop 5'//new_line('a')// &
        '  if (real(c8(2,2,2,2)) /= -7.0d0) error stop 6'//new_line('a')// &
        '  if (aimag(c8(2,2,2,2)) /= 8.0d0) error stop 7'//new_line('a')// &
        '  if (size(i4) /= 16) error stop 8'//new_line('a')// &
        '  if (lbound(i4,1) /= 1) error stop 9'//new_line('a')// &
        '  if (ubound(i4,4) /= 2) error stop 10'//new_line('a')// &
        '  print *, sum(i4), sum(r4), count(flags), real(c4(1,1,1,1)), '// &
        'aimag(c4(1,1,1,1)), real(c8(2,2,2,2)), aimag(c8(2,2,2,2)), '// &
        'size(i4)'//new_line('a')// &
        'end program main'

    print *, '=== direct session fixed module rank-4 compiler test ==='
    if (.not. expect_output_matches_gfortran(source, &
            'module_fixed_rank4')) stop 1
    print *, 'PASS: fixed rank-4 module arrays match independent checks and gfortran'
end program test_session_module_fixed_rank4_compiler
