program test_session_reject_io_01_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== I/O control-list constraint rejection compiler test ==='

    all_passed = .true.
    if (.not. test_nondefault_kind_spec_rejected()) all_passed = .false.
    if (.not. test_array_spec_rejected()) all_passed = .false.
    if (.not. test_end_in_output_rejected()) all_passed = .false.
    if (.not. test_write_without_unit_rejected()) all_passed = .false.
    if (.not. test_character_write_unit_rejected()) all_passed = .false.
    if (.not. test_negative_unit_rejected()) all_passed = .false.
    if (.not. test_newunit_without_file_rejected()) all_passed = .false.
    if (.not. test_boz_output_item_rejected()) all_passed = .false.
    if (.not. test_print_trailing_comma_rejected()) all_passed = .false.
    if (.not. test_unbalanced_parens_rejected()) all_passed = .false.
    if (.not. test_dec_specifier_rejected()) all_passed = .false.
    if (.not. test_valid_io_accepted()) all_passed = .false.
    if (.not. test_valid_open_newunit_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: invalid I/O control lists rejected, valid I/O accepted'

contains

    logical function test_nondefault_kind_spec_rejected()
        ! A character I/O specifier value must be a default-kind character
        ! string; char(1000, 4) is kind 4 (gfortran: "must be a character
        ! string of default kind").
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  open (1, blank=char(1000,4))'//new_line('a')// &
            'end program main'

        test_nondefault_kind_spec_rejected = expect_error_contains( &
            source, 'default kind', '/tmp/ffc_io01_kind_reject')
    end function test_nondefault_kind_spec_rejected

    logical function test_array_spec_rejected()
        ! An I/O specifier value must be scalar; an array constructor is not.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            "  read (1, asynchronous=['no'])"//new_line('a')// &
            'end program main'

        test_array_spec_rejected = expect_error_contains( &
            source, 'must be scalar', '/tmp/ffc_io01_scalar_reject')
    end function test_array_spec_rejected

    logical function test_end_in_output_rejected()
        ! END= is only meaningful for input; it is not allowed on WRITE.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  write (unit=6,end=999) 0'//new_line('a')// &
            '999 continue'//new_line('a')// &
            'end program main'

        test_end_in_output_rejected = expect_error_contains( &
            source, 'END=', '/tmp/ffc_io01_end_reject')
    end function test_end_in_output_rejected

    logical function test_write_without_unit_rejected()
        ! A WRITE control list must specify a unit.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            "  write(fmt='(a)'), 'abc'"//new_line('a')// &
            'end program main'

        test_write_without_unit_rejected = expect_error_contains( &
            source, 'UNIT not specified', '/tmp/ffc_io01_nounit_reject')
    end function test_write_without_unit_rejected

    logical function test_character_write_unit_rejected()
        ! A character literal cannot be a WRITE unit: an internal file must be
        ! a character variable, so write('(a)'), "x" is invalid.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            "  write ('(a)'), 'invalid'"//new_line('a')// &
            'end program main'

        test_character_write_unit_rejected = expect_error_contains( &
            source, 'Invalid form of WRITE statement', &
            '/tmp/ffc_io01_charunit_reject')
    end function test_character_write_unit_rejected

    logical function test_negative_unit_rejected()
        ! A unit number is never negative.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: file_exists'//new_line('a')// &
            '  inquire(unit=-1,exist=file_exists)'//new_line('a')// &
            'end program main'

        test_negative_unit_rejected = expect_error_contains( &
            source, 'cannot be negative', '/tmp/ffc_io01_negunit_reject')
    end function test_negative_unit_rejected

    logical function test_newunit_without_file_rejected()
        ! OPEN(NEWUNIT=...) requires FILE= or STATUS='SCRATCH'.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: lun'//new_line('a')// &
            '  open(newunit=lun)'//new_line('a')// &
            'end program main'

        test_newunit_without_file_rejected = expect_error_contains( &
            source, 'NEWUNIT specifier must have', &
            '/tmp/ffc_io01_newunit_reject')
    end function test_newunit_without_file_rejected

    logical function test_boz_output_item_rejected()
        ! A BOZ literal constant has no type and cannot be written out.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            "  print *, z'10110'"//new_line('a')// &
            'end program main'

        test_boz_output_item_rejected = expect_error_contains( &
            source, 'output IO list', '/tmp/ffc_io01_boz_reject')
    end function test_boz_output_item_rejected

    logical function test_print_trailing_comma_rejected()
        ! A comma after the format must be followed by an output item list.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *,'//new_line('a')// &
            'end program main'

        test_print_trailing_comma_rejected = expect_error_contains( &
            source, 'output item list', '/tmp/ffc_io01_comma_reject')
    end function test_print_trailing_comma_rejected

    logical function test_unbalanced_parens_rejected()
        ! An unbalanced parenthesis in a statement is never valid Fortran.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer(len((c)) :: n'//new_line('a')// &
            'end program main'

        test_unbalanced_parens_rejected = expect_error_contains( &
            source, 'unbalanced parenthes', '/tmp/ffc_io01_paren_reject')
    end function test_unbalanced_parens_rejected

    logical function test_dec_specifier_rejected()
        ! READONLY and friends are DEC extensions, not Fortran specifiers.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: fd'//new_line('a')// &
            '  fd = 10'//new_line('a')// &
            '  open (unit=fd, readonly)'//new_line('a')// &
            'end program main'

        test_dec_specifier_rejected = expect_error_contains( &
            source, 'DEC extension', '/tmp/ffc_io01_dec_reject')
    end function test_dec_specifier_rejected

    logical function test_valid_io_accepted()
        ! Corrected neighbours of every rejected form above still compile and
        ! run: default-kind character specifiers, scalar specifiers, a WRITE
        ! with a unit, a positive unit, and a typed output item.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: file_exists'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  character(len=20) :: buffer'//new_line('a')// &
            '  inquire(unit=6,exist=file_exists)'//new_line('a')// &
            "  write (6,'(a)') 'valid'"//new_line('a')// &
            "  write (buffer,'(a)') 'internal'"//new_line('a')// &
            '  i = 42'//new_line('a')// &
            '  print *, i'//new_line('a')// &
            "  print '(a)', 'done'"//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_valid_io_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_io01_valid_accept')
    end function test_valid_io_accepted

    logical function test_valid_open_newunit_accepted()
        ! OPEN with NEWUNIT and a FILE=, plus a default-kind character
        ! specifier, remains accepted.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: lun'//new_line('a')// &
            "  open(newunit=lun, file='/tmp/ffc_io01_scratch.txt', "// &
            "status='replace')"//new_line('a')// &
            "  close(lun, status='delete')"//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_valid_open_newunit_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_io01_newunit_accept')
    end function test_valid_open_newunit_accepted

end program test_session_reject_io_01_compiler
