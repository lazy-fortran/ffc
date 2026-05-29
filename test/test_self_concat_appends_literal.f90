program test_self_concat_appends_literal
    use ffc_test_support, only: expect_output
    implicit none

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

end program test_self_concat_appends_literal
