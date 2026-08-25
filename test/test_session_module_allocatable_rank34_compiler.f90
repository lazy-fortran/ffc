program test_session_module_allocatable_rank34_compiler
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none

    character(len=*), parameter :: source = &
        'module state'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  integer, allocatable, dimension(:,:,:), public :: i3'// &
        new_line('a')// &
        '  logical, allocatable, dimension(:,:,:), public :: flags3'// &
        new_line('a')// &
        '  real, allocatable, dimension(:,:,:,:), public :: r4'// &
        new_line('a')// &
        '  logical, allocatable, dimension(:,:,:,:), public :: flags4'// &
        new_line('a')// &
        'end module state'//new_line('a')// &
        'program main'//new_line('a')// &
        '  use state'//new_line('a')// &
        '  allocate(i3(2,2,2), flags3(2,2,2))'//new_line('a')// &
        '  allocate(r4(2,2,2,2), flags4(2,2,2,2))'//new_line('a')// &
        '  i3 = 4'//new_line('a')// &
        '  flags3 = .false.'//new_line('a')// &
        '  r4 = 1.5'//new_line('a')// &
        '  flags4 = .false.'//new_line('a')// &
        '  i3(2,1,2) = 17'//new_line('a')// &
        '  flags3(1,1,1) = .true.'//new_line('a')// &
        '  flags3(2,2,2) = .true.'//new_line('a')// &
        '  r4(2,2,2,2) = 9.5'//new_line('a')// &
        '  flags4(1,1,1,1) = .true.'//new_line('a')// &
        '  flags4(2,2,2,2) = .true.'//new_line('a')// &
        '  print *, sum(i3), flags3(1,1,1), flags3(2,2,2), size(i3,3)'// &
        new_line('a')// &
        '  print *, sum(r4), flags4(1,1,1,1), flags4(2,2,2,2), size(r4,4)'// &
        new_line('a')// &
        '  deallocate(i3, flags3, r4, flags4)'//new_line('a')// &
        'end program main'

    print *, '=== direct session module allocatable rank-3/rank-4 test ==='
    if (.not. expect_output_matches_gfortran(source, &
            'module_allocatable_rank34')) stop 1
    print *, 'PASS: rank-3 and rank-4 module allocatables match gfortran'
end program test_session_module_allocatable_rank34_compiler
