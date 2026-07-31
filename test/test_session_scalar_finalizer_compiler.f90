program test_session_scalar_finalizer_compiler
    ! Scalar FINAL procedures (#403): an applicable scalar finaliser runs
    ! exactly once when an owned local or allocatable derived value reaches
    ! the end of its lifetime, and never for a borrowed dummy.
    use ffc_test_support, only: expect_exit_status, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== scalar finalizer compiler test ==='

    all_passed = .true.
    if (.not. test_local_scope_exit_finalizes_once()) all_passed = .false.
    if (.not. test_deallocate_finalizes_once()) all_passed = .false.
    if (.not. test_borrowed_dummy_not_finalized()) all_passed = .false.
    if (.not. test_ambiguous_final_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: scalar finalizers lower through direct LIRIC'

contains

    logical function test_local_scope_exit_finalizes_once()
        ! A local derived value is finalized exactly once when the procedure
        ! that owns it completes execution.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: calls = 0'//new_line('a')// &
            '  type :: res_t'//new_line('a')// &
            '    integer :: id = 0'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    final :: res_final'//new_line('a')// &
            '  end type'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine res_final(self)'//new_line('a')// &
            '    type(res_t), intent(inout) :: self'//new_line('a')// &
            '    calls = calls + self%id'//new_line('a')// &
            '  end subroutine res_final'//new_line('a')// &
            '  subroutine owner()'//new_line('a')// &
            '    type(res_t) :: r'//new_line('a')// &
            '    r%id = 1'//new_line('a')// &
            '  end subroutine owner'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call owner()'//new_line('a')// &
            '  if (calls /= 1) stop 1'//new_line('a')// &
            '  call owner()'//new_line('a')// &
            '  if (calls /= 2) stop 2'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_local_scope_exit_finalizes_once = expect_exit_status( &
            source, 0, '/tmp/ffc_final_scope_test')
    end function test_local_scope_exit_finalizes_once

    logical function test_deallocate_finalizes_once()
        ! deallocate() of an owned allocatable scalar finalizes it once,
        ! before the storage is released.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: calls = 0'//new_line('a')// &
            '  type :: res_t'//new_line('a')// &
            '    integer :: id = 0'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    final :: res_final'//new_line('a')// &
            '  end type'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine res_final(self)'//new_line('a')// &
            '    type(res_t), intent(inout) :: self'//new_line('a')// &
            '    calls = calls + 1'//new_line('a')// &
            '  end subroutine res_final'//new_line('a')// &
            '  subroutine owner()'//new_line('a')// &
            '    type(res_t), allocatable :: r'//new_line('a')// &
            '    allocate(r)'//new_line('a')// &
            '    r%id = 4'//new_line('a')// &
            '    deallocate(r)'//new_line('a')// &
            '  end subroutine owner'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call owner()'//new_line('a')// &
            '  if (calls /= 1) stop 1'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_deallocate_finalizes_once = expect_exit_status( &
            source, 0, '/tmp/ffc_final_dealloc_test')
    end function test_deallocate_finalizes_once

    logical function test_borrowed_dummy_not_finalized()
        ! A dummy argument is borrowed, not owned: returning from the callee
        ! must not finalize it. Only the owning scope finalizes.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: calls = 0'//new_line('a')// &
            '  type :: res_t'//new_line('a')// &
            '    integer :: id = 0'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    final :: res_final'//new_line('a')// &
            '  end type'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine res_final(self)'//new_line('a')// &
            '    type(res_t), intent(inout) :: self'//new_line('a')// &
            '    calls = calls + 1'//new_line('a')// &
            '  end subroutine res_final'//new_line('a')// &
            '  subroutine borrow(item)'//new_line('a')// &
            '    type(res_t), intent(inout) :: item'//new_line('a')// &
            '    item%id = 9'//new_line('a')// &
            '  end subroutine borrow'//new_line('a')// &
            '  subroutine owner()'//new_line('a')// &
            '    type(res_t) :: r'//new_line('a')// &
            '    call borrow(r)'//new_line('a')// &
            '  end subroutine owner'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call owner()'//new_line('a')// &
            '  if (calls /= 1) stop 1'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_borrowed_dummy_not_finalized = expect_exit_status( &
            source, 0, '/tmp/ffc_final_dummy_test')
    end function test_borrowed_dummy_not_finalized

    logical function test_ambiguous_final_rejected()
        ! Two scalar FINAL bindings of the same rank are ambiguous and stay
        ! rejected rather than silently picking one.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: res_t'//new_line('a')// &
            '    integer :: id = 0'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    final :: res_final_a'//new_line('a')// &
            '    final :: res_final_b'//new_line('a')// &
            '  end type'//new_line('a')// &
            '  type(res_t) :: r'//new_line('a')// &
            '  r%id = 3'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine res_final_a(self)'//new_line('a')// &
            '    type(res_t), intent(inout) :: self'//new_line('a')// &
            '    self%id = 1'//new_line('a')// &
            '  end subroutine res_final_a'//new_line('a')// &
            '  subroutine res_final_b(self)'//new_line('a')// &
            '    type(res_t), intent(inout) :: self'//new_line('a')// &
            '    self%id = 2'//new_line('a')// &
            '  end subroutine res_final_b'//new_line('a')// &
            'end program main'

        test_ambiguous_final_rejected = expect_error_contains( &
            source, 'final', '/tmp/ffc_final_ambiguous_test')
    end function test_ambiguous_final_rejected

end program test_session_scalar_finalizer_compiler
