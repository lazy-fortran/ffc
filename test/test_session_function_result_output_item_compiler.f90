program test_session_function_result_output_item_compiler
    ! A contained function's RESULT variable is an ordinary output-list item.
    ! An unrelated external procedure name remains an invalid bare item.
    use ffc_test_support, only: expect_output_matches_gfortran, &
        expect_error_contains
    implicit none
    character(len=*), parameter :: positive_source = &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  integer :: x'//new_line('a')// &
        '  x = f()'//new_line('a')// &
        '  print *, x'//new_line('a')// &
        'contains'//new_line('a')// &
        '  integer function f() result(r)'//new_line('a')// &
        '    r = 41'//new_line('a')// &
        '    write (*, *) r'//new_line('a')// &
        '  end function f'//new_line('a')// &
        'end program main'
    character(len=*), parameter :: negative_source = &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  external :: ext'//new_line('a')// &
        '  print *, ext'//new_line('a')// &
        'end program main'

    print *, '=== function result output-item compiler test ==='
    if (.not. expect_output_matches_gfortran(positive_source, &
            'function_result_output_item')) stop 1
    if (.not. expect_error_contains(negative_source, &
            'integer identifier was not declared: ext', &
            '/var/tmp/ert/ffc_issue581_external_output_item')) stop 1
    print *, 'PASS: result output item accepted; external name rejected'
end program test_session_function_result_output_item_compiler
