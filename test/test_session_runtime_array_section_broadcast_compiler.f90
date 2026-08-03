program test_session_runtime_array_section_broadcast_compiler
    use ffc_test_support, only: expect_output
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  call work(4)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    integer :: v(n), m(n,2)'//new_line('a')// &
        '    v = 1'//new_line('a')// &
        '    v(2:n) = 0'//new_line('a')// &
        '    m = 1'//new_line('a')// &
        '    m(:,2) = 2'//new_line('a')// &
        '    print *, v'//new_line('a')// &
        '    print *, m'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'
    character(len=*), parameter :: expected = &
        '           1           0           0           0'//new_line('a')// &
        '           1           1           1           1           2           2           2           2'// &
        new_line('a')

    print *, '=== direct session runtime array-section broadcast test ==='
    if (.not. expect_output(source, expected, &
            '/tmp/ffc_session_runtime_array_section_broadcast')) stop 1
    print *, 'PASS: runtime array-section scalar broadcast preserves section shape'
end program test_session_runtime_array_section_broadcast_compiler
