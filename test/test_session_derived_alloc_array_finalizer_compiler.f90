program test_session_derived_alloc_array_finalizer_compiler
    ! Finalization of allocatable derived arrays (#643 / #403): the type's
    ! scalar FINAL procedure runs once on every element of an allocated
    ! rank-1/rank-2 derived array, both when `deallocate` releases the block
    ! and when the owning procedure reaches the end of its scope. An
    ! unallocated array finalizes nothing.
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== derived allocatable array finalizer compiler test ==='

    all_passed = .true.
    if (.not. test_deallocate_finalizes_rank1()) all_passed = .false.
    if (.not. test_scope_exit_finalizes_rank1()) all_passed = .false.
    if (.not. test_rank2_finalizes_all()) all_passed = .false.
    if (.not. test_unallocated_finalizes_nothing()) all_passed = .false.
    if (.not. test_polymorphic_array_refused()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: allocatable derived array finalizers lower through direct LIRIC'

contains

    logical function test_deallocate_finalizes_rank1()
        ! deallocate(a) of an allocated rank-1 derived array runs the scalar
        ! finaliser once per element, in order, before the storage is released.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: sum_id = 0'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer :: id = 0'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    final :: box_final'//new_line('a')// &
            '  end type'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine box_final(self)'//new_line('a')// &
            '    type(box_t), intent(inout) :: self'//new_line('a')// &
            '    sum_id = sum_id + self%id'//new_line('a')// &
            '  end subroutine box_final'//new_line('a')// &
            '  subroutine owner()'//new_line('a')// &
            '    type(box_t), allocatable :: a(:)'//new_line('a')// &
            '    allocate(a(3))'//new_line('a')// &
            '    a(1)%id = 10'//new_line('a')// &
            '    a(2)%id = 20'//new_line('a')// &
            '    a(3)%id = 30'//new_line('a')// &
            '    deallocate(a)'//new_line('a')// &
            '  end subroutine owner'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call owner()'//new_line('a')// &
            '  if (sum_id /= 60) stop 1'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_deallocate_finalizes_rank1 = expect_exit_status( &
            source, 0, '/tmp/ffc_alloc_arr_final_dealloc')
    end function test_deallocate_finalizes_rank1

    logical function test_scope_exit_finalizes_rank1()
        ! An allocatable derived array that is still allocated when the owning
        ! procedure returns is finalized once per element at scope exit.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: n_final = 0'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer :: id = 0'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    final :: box_final'//new_line('a')// &
            '  end type'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine box_final(self)'//new_line('a')// &
            '    type(box_t), intent(inout) :: self'//new_line('a')// &
            '    n_final = n_final + 1'//new_line('a')// &
            '  end subroutine box_final'//new_line('a')// &
            '  subroutine owner()'//new_line('a')// &
            '    type(box_t), allocatable :: a(:)'//new_line('a')// &
            '    allocate(a(4))'//new_line('a')// &
            '    a(1)%id = 1'//new_line('a')// &
            '    a(2)%id = 2'//new_line('a')// &
            '    a(3)%id = 3'//new_line('a')// &
            '    a(4)%id = 4'//new_line('a')// &
            '  end subroutine owner'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call owner()'//new_line('a')// &
            '  if (n_final /= 4) stop 1'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_scope_exit_finalizes_rank1 = expect_exit_status( &
            source, 0, '/tmp/ffc_alloc_arr_final_scope')
    end function test_scope_exit_finalizes_rank1

    logical function test_rank2_finalizes_all()
        ! A rank-2 allocatable derived array finalizes every element: the
        ! finaliser runs once per element (8 for a 2x4 allocation).
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: n_final = 0'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer :: id = 0'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    final :: box_final'//new_line('a')// &
            '  end type'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine box_final(self)'//new_line('a')// &
            '    type(box_t), intent(inout) :: self'//new_line('a')// &
            '    n_final = n_final + 1'//new_line('a')// &
            '  end subroutine box_final'//new_line('a')// &
            '  subroutine owner()'//new_line('a')// &
            '    type(box_t), allocatable :: a(:,:)'//new_line('a')// &
            '    allocate(a(2,4))'//new_line('a')// &
            '    a(1,1)%id = 1'//new_line('a')// &
            '    deallocate(a)'//new_line('a')// &
            '  end subroutine owner'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call owner()'//new_line('a')// &
            '  if (n_final /= 8) stop 1'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_rank2_finalizes_all = expect_exit_status( &
            source, 0, '/tmp/ffc_alloc_arr_final_rank2')
    end function test_rank2_finalizes_all

    logical function test_unallocated_finalizes_nothing()
        ! An allocatable derived array that was never allocated, or was already
        ! deallocated, runs no finaliser.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: n_final = 0'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer :: id = 0'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    final :: box_final'//new_line('a')// &
            '  end type'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine box_final(self)'//new_line('a')// &
            '    type(box_t), intent(inout) :: self'//new_line('a')// &
            '    n_final = n_final + 1'//new_line('a')// &
            '  end subroutine box_final'//new_line('a')// &
            '  subroutine owner()'//new_line('a')// &
            '    type(box_t), allocatable :: a(:)'//new_line('a')// &
            '    if (allocated(a)) stop 1'//new_line('a')// &
            '  end subroutine owner'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call owner()'//new_line('a')// &
            '  if (n_final /= 0) stop 1'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_unallocated_finalizes_nothing = expect_exit_status( &
            source, 0, '/tmp/ffc_alloc_arr_final_unalloc')
    end function test_unallocated_finalizes_nothing

    logical function test_polymorphic_array_refused()
        ! The current finalizer loop is monomorphic. A polymorphic array must
        ! fail before deallocation rather than use its declared type's layout.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    final :: base_final'//new_line('a')// &
            '  end type'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine base_final(self)'//new_line('a')// &
            '    type(base_t), intent(inout) :: self'//new_line('a')// &
            '  end subroutine base_final'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  class(base_t), allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(2))'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_polymorphic_array_refused = expect_error_contains( &
            source, 'polymorphic allocatable array finalization is not supported', &
            '/tmp/ffc_alloc_arr_final_polymorphic')
    end function test_polymorphic_array_refused

end program test_session_derived_alloc_array_finalizer_compiler
