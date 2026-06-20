program test_session_formatted_read_compiler
    ! Formatted (non-list-directed) READ from a file unit: write values with
    ! explicit edit descriptors, rewind, read them back, and print.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session formatted-read compiler test ==='

    all_passed = .true.
    if (.not. test_formatted_int_and_real_read()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: formatted file-unit read lowers through direct LIRIC session'

contains

    logical function test_formatted_int_and_real_read()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u, v'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  open(newunit=u, status='//achar(39)//'scratch'//achar(39)//')'// &
            new_line('a')// &
            '  write(u, '//achar(39)//'(I5)'//achar(39)//') 42'//new_line('a')// &
            '  write(u, '//achar(39)//'(F8.3)'//achar(39)//') 3.14'// &
            new_line('a')// &
            '  rewind(u)'//new_line('a')// &
            '  read(u, '//achar(39)//'(I5)'//achar(39)//') v'//new_line('a')// &
            '  read(u, '//achar(39)//'(F8.3)'//achar(39)//') r'//new_line('a')// &
            '  print *, v'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            '  close(u)'//new_line('a')// &
            'end program main'

        test_formatted_int_and_real_read = expect_output(source, &
            repeat(' ', 10)//'42'//new_line('a')// &
            '   3.14000010    '//new_line('a'), &
            '/tmp/ffc_formatted_read')
    end function test_formatted_int_and_real_read

end program test_session_formatted_read_compiler
