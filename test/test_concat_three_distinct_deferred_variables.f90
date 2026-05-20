program test_concat_three_distinct_deferred_variables
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== three distinct deferred concat test ==='

    all_passed = .true.
    if (.not. test_distinct_pair()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: three distinct deferred char concat'

contains

    logical function test_distinct_pair()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: a'//new_line('a')// &
            '  character(len=:), allocatable :: b'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  a = "he"'//new_line('a')// &
            '  b = "llo"'//new_line('a')// &
            '  s = a // b'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_distinct_pair = expect_output( &
            source, 'hello'//new_line('a'), &
            '/tmp/ffc_distinct_deferred_concat_test')
    end function test_distinct_pair

end program test_concat_three_distinct_deferred_variables
