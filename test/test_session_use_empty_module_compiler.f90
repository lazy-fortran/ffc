! fo: dispatcher
module ffc_case_test_session_use_empty_module_compiler
    implicit none
    private
    public :: case_test_session_use_empty_module_compiler
    interface
        subroutine case_test_session_use_empty_module_compiler()
        end subroutine case_test_session_use_empty_module_compiler
    end interface
end module ffc_case_test_session_use_empty_module_compiler

subroutine case_test_session_use_empty_module_compiler()
    use ffc_test_support, only: expect_exit_status
    implicit none
    save

    logical :: all_passed

    print *, '=== direct session use-empty-module compiler test ==='

    all_passed = .true.
    if (.not. test_use_empty_module_compiles()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: use of an empty module compiles and runs'

contains

    logical function test_use_empty_module_compiles()
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 7'//new_line('a')// &
            '  stop x'//new_line('a')// &
            'end program main'

        test_use_empty_module_compiles = expect_exit_status( &
            source, 7, '/tmp/ffc_session_use_empty_module_test')
    end function test_use_empty_module_compiles

end subroutine case_test_session_use_empty_module_compiler
