! fo: dispatcher
module ffc_case_test_self_concat_three_times
    implicit none
    private
    public :: case_test_self_concat_three_times
    interface
        subroutine case_test_self_concat_three_times()
        end subroutine case_test_self_concat_three_times
    end interface
end module ffc_case_test_self_concat_three_times

subroutine case_test_self_concat_three_times()
    use ffc_test_support, only: expect_output
    implicit none
    save

    logical :: all_passed

    print *, '=== self-aliasing triple concat test ==='

    all_passed = .true.
    if (.not. test_self_concat_three_times_case()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: triple self-aliasing deferred char concat'

contains

    logical function test_self_concat_three_times_case()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  s = "x"'//new_line('a')// &
            '  s = s // "."'//new_line('a')// &
            '  s = s // "."'//new_line('a')// &
            '  s = s // "."'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_self_concat_three_times_case = expect_output( &
            source, ' x...'//new_line('a'), &
            '/tmp/ffc_self_concat_triple_test')
    end function test_self_concat_three_times_case

end subroutine case_test_self_concat_three_times
