! fo: dispatcher
module ffc_case_test_self_concat_appends_literal
    implicit none
    private
    public :: case_test_self_concat_appends_literal
    interface
        subroutine case_test_self_concat_appends_literal()
        end subroutine case_test_self_concat_appends_literal
    end interface
end module ffc_case_test_self_concat_appends_literal

subroutine case_test_self_concat_appends_literal()
    use ffc_test_support, only: expect_output
    implicit none
    save

    logical :: all_passed

    print *, '=== self-aliasing concat test ==='

    all_passed = .true.
    if (.not. test_self_concat_appends_literal_case()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: self-aliasing deferred char concat'

contains

    logical function test_self_concat_appends_literal_case()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  s = "hi"'//new_line('a')// &
            '  s = s // "!"'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_self_concat_appends_literal_case = expect_output( &
            source, ' hi!'//new_line('a'), &
            '/tmp/ffc_self_concat_literal_test')
    end function test_self_concat_appends_literal_case

end subroutine case_test_self_concat_appends_literal
