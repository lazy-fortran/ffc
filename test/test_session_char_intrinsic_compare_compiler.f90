program test_session_char_intrinsic_compare_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session char() intrinsic comparison test ==='

    all_passed = .true.
    if (.not. test_char_ordering()) all_passed = .false.
    if (.not. test_char_print()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: char() intrinsic lowers as a character expression'

contains

    logical function test_char_ordering()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  if (.not. (char(0) < char(255))) error stop'//new_line('a')// &
            '  if (.not. (char(127) < char(128))) error stop'//new_line('a')// &
            '  if (.not. (char(65) == "A")) error stop'//new_line('a')// &
            '  if (char(200) <= char(100)) error stop'//new_line('a')// &
            'end program main'

        test_char_ordering = expect_exit_status( &
            source, 0, '/tmp/ffc_session_char_intrin_ord')
    end function test_char_ordering

    logical function test_char_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, char(65), char(66)'//new_line('a')// &
            'end program main'

        test_char_print = expect_output( &
            source, ' AB'//new_line('a'), &
            '/tmp/ffc_session_char_intrin_print')
    end function test_char_print

end program test_session_char_intrinsic_compare_compiler
