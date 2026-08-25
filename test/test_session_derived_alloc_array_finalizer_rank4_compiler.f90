program test_session_derived_alloc_array_finalizer_rank4_compiler
    ! Rank-4 allocatable derived arrays must allocate all four extents and run
    ! one scalar FINAL procedure per element on deallocation and scope exit.
    use ffc_test_support, only: expect_exit_status
    implicit none

    character(len=*), parameter :: source = &
        'module m'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  integer :: n_final = 0'//new_line('a')// &
        '  type :: box_t'//new_line('a')// &
        '  contains'//new_line('a')// &
        '    final :: box_final'//new_line('a')// &
        '  end type'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine box_final(self)'//new_line('a')// &
        '    type(box_t), intent(inout) :: self'//new_line('a')// &
        '    n_final = n_final + 1'//new_line('a')// &
        '  end subroutine box_final'//new_line('a')// &
        '  subroutine explicit_deallocate()'//new_line('a')// &
        '    type(box_t), allocatable :: a(:,:,:,:)'//new_line('a')// &
        '    allocate(a(2,3,1,2))'//new_line('a')// &
        '    deallocate(a)'//new_line('a')// &
        '  end subroutine explicit_deallocate'//new_line('a')// &
        '  subroutine scope_exit()'//new_line('a')// &
        '    type(box_t), allocatable :: a(:,:,:,:)'//new_line('a')// &
        '    allocate(a(1,2,2,2))'//new_line('a')// &
        '  end subroutine scope_exit'//new_line('a')// &
        'end module m'//new_line('a')// &
        'program main'//new_line('a')// &
        '  use m'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  call explicit_deallocate()'//new_line('a')// &
        '  if (n_final /= 12) stop 1'//new_line('a')// &
        '  call scope_exit()'//new_line('a')// &
        '  if (n_final /= 20) stop 1'//new_line('a')// &
        '  stop 0'//new_line('a')// &
        'end program main'

    if (.not. expect_exit_status(source, 0, &
            '/tmp/ffc_alloc_arr_final_rank4')) stop 1
    print *, 'PASS: rank-4 derived allocatable finalizer count oracle'
end program test_session_derived_alloc_array_finalizer_rank4_compiler
