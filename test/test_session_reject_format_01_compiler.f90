program test_session_reject_format_01_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== malformed FORMAT descriptor rejection compiler test ==='

    all_passed = .true.
    if (.not. test_noncharacter_format_tag_rejected()) all_passed = .false.
    if (.not. test_character_format_tag_accepted()) all_passed = .false.
    if (.not. test_zero_scale_factor_rejected()) all_passed = .false.
    if (.not. test_p_edit_descriptor_accepted()) all_passed = .false.
    if (.not. test_nonconstant_asynchronous_rejected()) all_passed = .false.
    if (.not. test_constant_asynchronous_accepted()) all_passed = .false.
    if (.not. test_unbalanced_format_string_rejected()) all_passed = .false.
    if (.not. test_balanced_format_string_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: malformed FORMAT descriptors rejected, corrected ' // &
        'neighbours still accepted'

contains

    logical function test_noncharacter_format_tag_rejected()
        ! R1: a format tag that names an entity must be a default character
        ! entity or a statement label; a REAL named constant is neither
        ! (gfortran: "Invalid expression in the FORMAT tag").
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real, parameter :: a = 1.'//new_line('a')// &
            '  write(*,a) "test"'//new_line('a')// &
            'end program main'

        test_noncharacter_format_tag_rejected = expect_error_contains( &
            source, 'Invalid expression in the FORMAT tag', &
            '/tmp/ffc_session_format_tag_reject')
    end function test_noncharacter_format_tag_rejected

    logical function test_character_format_tag_accepted()
        ! Corrected neighbour of R1: the same write with a CHARACTER named
        ! constant as format tag compiles and runs.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=*), parameter :: a = "(A)"'//new_line('a')// &
            '  write(*,a) "test"'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_character_format_tag_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_format_tag_accept')
    end function test_character_format_tag_accepted

    logical function test_zero_scale_factor_rejected()
        ! R2: a leading zero repeat specification is only well formed as the
        ! scale factor of a P edit descriptor; 0F9.4 has a zero repeat count
        ! (gfortran: "Expected P edit descriptor").
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '2050  format(0F9.4)'//new_line('a')// &
            'end program main'

        test_zero_scale_factor_rejected = expect_error_contains( &
            source, 'Expected P edit descriptor', &
            '/tmp/ffc_session_format_zero_reject')
    end function test_zero_scale_factor_rejected

    logical function test_p_edit_descriptor_accepted()
        ! Corrected neighbour of R2: the zero is a scale factor for P.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '2050  format(0PF9.4)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_p_edit_descriptor_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_format_zero_accept')
    end function test_p_edit_descriptor_accepted

    logical function test_nonconstant_asynchronous_rejected()
        ! R3: the ASYNCHRONOUS= specifier of a data transfer statement must be
        ! an initialization expression, so a reference to a non-intrinsic
        ! function is invalid (gfortran: "must be an intrinsic function").
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  write(*,*,asynchronous=no()) 1'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function no()'//new_line('a')// &
            '    character(3) :: no'//new_line('a')// &
            '    no = "yes"'//new_line('a')// &
            '  end function no'//new_line('a')// &
            'end program main'

        test_nonconstant_asynchronous_rejected = expect_error_contains( &
            source, 'must be an intrinsic function', &
            '/tmp/ffc_session_format_async_reject')
    end function test_nonconstant_asynchronous_rejected

    logical function test_constant_asynchronous_accepted()
        ! Corrected neighbour of R3: a character literal ASYNCHRONOUS= value is
        ! an initialization expression and stays accepted.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  write(*,*,asynchronous="no") 1'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_constant_asynchronous_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_format_async_accept')
    end function test_constant_asynchronous_accepted

    logical function test_unbalanced_format_string_rejected()
        ! R4: a format built by concatenating character literals must still be
        ! a complete format specification; "((" // "A)" leaves an unclosed
        ! left parenthesis (gfortran: "Unexpected end of format string").
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(10) :: var'//new_line('a')// &
            '  read (''(('') // ''A)'', var'//new_line('a')// &
            'end program main'

        test_unbalanced_format_string_rejected = expect_error_contains( &
            source, 'Unexpected end of format string', &
            '/tmp/ffc_session_format_paren_reject')
    end function test_unbalanced_format_string_rejected

    logical function test_balanced_format_string_accepted()
        ! Corrected neighbour of R4: the same concatenation closing its
        ! parenthesis is a complete format specification.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(10) :: var'//new_line('a')// &
            '  var = "x"'//new_line('a')// &
            '  read (''(A'') // '')'', var'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_balanced_format_string_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_format_paren_accept')
    end function test_balanced_format_string_accepted

end program test_session_reject_format_01_compiler
