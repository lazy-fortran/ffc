program test_session_logical_result_call_compiler
    ! Regression for #576.  This is the standard-Fortran equivalent emitted
    ! from issue_2064's Lazy source: a contained logical function result is
    ! used in an assignment after a compound comparison.
    use ffc_test_support, only: expect_output
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer :: x'//new_line('a')// &
        '  logical :: result, result2'//new_line('a')// &
        '  x = 5'//new_line('a')// &
        '  result = is_in_range(x, 1, 10)'//new_line('a')// &
        '  print *, ''In range:'', result'//new_line('a')// &
        '  x = 15'//new_line('a')// &
        '  result2 = is_in_range(x, 1, 10)'//new_line('a')// &
        '  print *, ''Out of range:'', result2'//new_line('a')// &
        'contains'//new_line('a')// &
        '  logical function is_in_range(val, min_val, max_val)'//new_line('a')// &
        '    integer, intent(in) :: val, min_val, max_val'//new_line('a')// &
        '    is_in_range = (val >= min_val) .and. (val <= max_val)'//new_line('a')// &
        '  end function is_in_range'//new_line('a')// &
        'end program main'

    print *, '=== logical function result call compiler test ==='
    if (.not. expect_output(source, &
            ' In range: T'//new_line('a')// &
            ' Out of range: F'//new_line('a'), &
            '/var/tmp/ert/ffc_issue576_logical_result_call')) stop 1
    print *, 'PASS: contained logical function results lower without a crash'
end program test_session_logical_result_call_compiler
