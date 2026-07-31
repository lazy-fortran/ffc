program test_session_multi_allocate
    ! Multi-variable allocate/deallocate and rank-2 scalar broadcast:
    !   allocate(a(N), b(N), c(N), stat=ierr)
    !   deallocate(a, b, c)
    !   allocate(m(P,Q)); m = scalar   (rank-2 whole-array broadcast)
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session multi-variable allocate compiler test ==='

    all_passed = .true.
    if (.not. test_multi_var_allocate()) all_passed = .false.
    if (.not. test_multi_var_deallocate()) all_passed = .false.
    if (.not. test_rank2_broadcast()) all_passed = .false.
    if (.not. test_distinct_extents()) all_passed = .false.
    if (.not. test_module_allocatable_multi()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: multi-variable allocate and rank-2 broadcast lower through LIRIC'

contains

    logical function test_multi_var_allocate()
        ! allocate three rank-1 arrays in one statement, fill and sum; stat=0.
        ! a(1)+a(2)+a(3) = 15, b(2) = 7, so 15 + 7 + ierr(0) = 22.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:), b(:), c(:)'//new_line('a')// &
            '  integer :: ierr'//new_line('a')// &
            '  allocate(a(3), b(3), c(3), stat=ierr)'//new_line('a')// &
            '  a = 5'//new_line('a')// &
            '  b = 7'//new_line('a')// &
            '  c = 1'//new_line('a')// &
            '  stop a(1) + a(2) + a(3) + b(2) + ierr'//new_line('a')// &
            'end program main'

        test_multi_var_allocate = expect_exit_status( &
            source, 22, '/tmp/ffc_multi_alloc')
    end function test_multi_var_allocate

    logical function test_multi_var_deallocate()
        ! allocate two arrays, then deallocate both in one statement; the sizes
        ! were 4 and 4, so stop with size(a)+size(b) before deallocation = 8.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:), b(:)'//new_line('a')// &
            '  integer :: total'//new_line('a')// &
            '  allocate(a(4), b(4))'//new_line('a')// &
            '  total = size(a) + size(b)'//new_line('a')// &
            '  deallocate(a, b)'//new_line('a')// &
            '  stop total'//new_line('a')// &
            'end program main'

        test_multi_var_deallocate = expect_exit_status( &
            source, 8, '/tmp/ffc_multi_dealloc')
    end function test_multi_var_deallocate

    logical function test_rank2_broadcast()
        ! rank-2 allocatable whole-array scalar broadcast: m = 9 fills every
        ! element; read back two corners: 9 + 9 = 18.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: m(:,:)'//new_line('a')// &
            '  allocate(m(3,4))'//new_line('a')// &
            '  m = 9'//new_line('a')// &
            '  stop m(1,1) + m(3,4)'//new_line('a')// &
            'end program main'

        test_rank2_broadcast = expect_exit_status( &
            source, 18, '/tmp/ffc_rank2_broadcast')
    end function test_rank2_broadcast

    logical function test_distinct_extents()
        ! Each target of a multi-object ALLOCATE uses its own shape expression:
        ! 555 and 1000. stop size(a)/5 + size(b)/100 = 111 + 10 = 121.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:), b(:)'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  n = 555'//new_line('a')// &
            '  allocate(a(n), b(1000))'//new_line('a')// &
            '  stop size(a) / 5 + size(b) / 100'//new_line('a')// &
            'end program main'

        test_distinct_extents = expect_exit_status( &
            source, 121, '/tmp/ffc_multi_alloc_extents')
    end function test_distinct_extents

    logical function test_module_allocatable_multi()
        ! Module-scope allocatable arrays allocated with distinct extents in one
        ! statement (lfortran modules_58_module.f90 shape). Both descriptors are
        ! module globals, so the sizes survive the return to the program.
        character(len=*), parameter :: source = &
            'module m58'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, public :: nx1 = 1000'//new_line('a')// &
            '  integer, public :: nx2 = 1000'//new_line('a')// &
            '  real, allocatable, dimension(:), public :: ref_flux'//new_line('a')// &
            '  real, allocatable, dimension(:), public :: ref_fluxm'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine mms_allocate()'//new_line('a')// &
            '    nx1 = 555'//new_line('a')// &
            '    allocate(ref_flux(nx1), ref_fluxm(nx2))'//new_line('a')// &
            '    if (size(ref_flux) /= 555) error stop'//new_line('a')// &
            '    if (size(ref_fluxm) /= 1000) error stop'//new_line('a')// &
            '  end subroutine mms_allocate'//new_line('a')// &
            'end module m58'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m58'//new_line('a')// &
            '  call mms_allocate()'//new_line('a')// &
            '  stop size(ref_flux) / 5 + size(ref_fluxm) / 100'//new_line('a')// &
            'end program main'

        test_module_allocatable_multi = expect_exit_status( &
            source, 121, '/tmp/ffc_module_multi_alloc')
    end function test_module_allocatable_multi

end program test_session_multi_allocate
