! fo: dispatcher
module ffc_case_test_session_allocated_keyword_compiler
    implicit none
    private
    public :: case_test_session_allocated_keyword_compiler
    interface
        subroutine case_test_session_allocated_keyword_compiler()
        end subroutine case_test_session_allocated_keyword_compiler
    end interface
end module ffc_case_test_session_allocated_keyword_compiler

subroutine case_test_session_allocated_keyword_compiler()
    ! Behavioral oracle for ALLOCATED keyword arguments: both the scalar and
    ! array inquiry forms must reflect their allocation state at run time.
    use ffc_test_support, only: expect_exit_status
    implicit none
    save

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  integer, allocatable :: x'//new_line('a')// &
        '  integer, allocatable :: a(:)'//new_line('a')// &
        '  if (allocated(scalar=x)) error stop 1'//new_line('a')// &
        '  if (allocated(array=a)) error stop 2'//new_line('a')// &
        '  allocate(x)'//new_line('a')// &
        '  allocate(a(2))'//new_line('a')// &
        '  if (.not. allocated(scalar=x)) error stop 3'//new_line('a')// &
        '  if (.not. allocated(array=a)) error stop 4'//new_line('a')// &
        '  stop 0'//new_line('a')// &
        'end program main'

    print *, '=== direct session allocated keyword compiler test ==='
    if (.not. expect_exit_status(source, 0, &
        '/tmp/ffc_allocated_keyword_test')) stop 1
    print *, 'PASS: ALLOCATED keyword arguments lower behaviorally'
end subroutine case_test_session_allocated_keyword_compiler
