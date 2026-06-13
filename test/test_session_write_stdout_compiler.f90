program test_session_write_stdout_compiler
    ! write(*,*) and write(*,'(fmt)') are aliases for print * and print '(fmt)'.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session write(*,...) compiler test ==='

    all_passed = .true.
    if (.not. test_write_star_star_integer()) all_passed = .false.
    if (.not. test_write_star_fmt_integer()) all_passed = .false.
    if (.not. test_write_6_star_integer()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: write(*,...) lowers through direct LIRIC session'

contains

    logical function test_write_star_star_integer()
        ! write(*,*) x  ->  same output as  print *, x
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 7'//new_line('a')// &
            '  write (*, *) x'//new_line('a')// &
            'end program main'

        test_write_star_star_integer = expect_output(source, &
            '           7'//new_line('a'), '/tmp/ffc_write_star_star_test')
    end function test_write_star_star_integer

    logical function test_write_star_fmt_integer()
        ! write(*,'(I0)') x  ->  same output as  print '(I0)', x
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 42'//new_line('a')// &
            "  write (*, '(I0)') x"//new_line('a')// &
            'end program main'

        test_write_star_fmt_integer = expect_output(source, &
            '42'//new_line('a'), '/tmp/ffc_write_star_fmt_test')
    end function test_write_star_fmt_integer

    logical function test_write_6_star_integer()
        ! Unit 6 is the standard-output unit in Fortran.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 99'//new_line('a')// &
            '  write (6, *) x'//new_line('a')// &
            'end program main'

        test_write_6_star_integer = expect_output(source, &
            '          99'//new_line('a'), '/tmp/ffc_write_6_star_test')
    end function test_write_6_star_integer

end program test_session_write_stdout_compiler
