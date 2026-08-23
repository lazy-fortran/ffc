program test_session_file_write_character_literal_compiler
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    print *, '=== file WRITE character literal compiler test ==='

    all_passed = .true.
    if (.not. test_list_directed_literal()) all_passed = .false.
    if (.not. test_explicit_format_control()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: file WRITE character literal matches gfortran'

contains

    logical function test_list_directed_literal()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u'//new_line('a')// &
            '  character(len=8) :: line'//new_line('a')// &
            "  open(newunit=u, status='scratch')"//new_line('a')// &
            "  write(u, *) 'foo'"//new_line('a')// &
            '  rewind(u)'//new_line('a')// &
            "  read(u, '(A)') line"//new_line('a')// &
            '  close(u)'//new_line('a')// &
            "  print '(A)', line"//new_line('a')// &
            'end program main'

        test_list_directed_literal = expect_output_matches_gfortran(source, &
            'file_write_character_literal')
    end function test_list_directed_literal

    logical function test_explicit_format_control()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u'//new_line('a')// &
            '  character(len=8) :: line'//new_line('a')// &
            "  open(newunit=u, status='scratch')"//new_line('a')// &
            "  write(u, '(A)') 'foo'"//new_line('a')// &
            '  rewind(u)'//new_line('a')// &
            "  read(u, '(A)') line"//new_line('a')// &
            '  close(u)'//new_line('a')// &
            "  print '(A)', line"//new_line('a')// &
            'end program main'

        test_explicit_format_control = expect_output_matches_gfortran(source, &
            'file_write_character_format_control')
    end function test_explicit_format_control

end program test_session_file_write_character_literal_compiler
