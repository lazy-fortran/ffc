program test_session_reject_charlen_01_compiler
    ! #384: a character length specification must be a scalar INTEGER
    ! expression. Each invalid literal form is rejected with a source
    ! diagnostic, while the corrected integer neighbour still compiles and
    ! runs.
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    character(len=*), parameter :: FRAGMENT = &
        'character length must be a scalar INTEGER expression'
    logical :: all_passed

    print *, '=== character length expression rejection test ==='

    all_passed = .true.
    if (.not. test_real_length_rejected()) all_passed = .false.
    if (.not. test_double_length_rejected()) all_passed = .false.
    if (.not. test_complex_length_rejected()) all_passed = .false.
    if (.not. test_logical_length_rejected()) all_passed = .false.
    if (.not. test_character_component_length_rejected()) all_passed = .false.
    if (.not. test_implicit_nonconstant_length_rejected()) all_passed = .false.
    if (.not. test_integer_length_accepted()) all_passed = .false.
    if (.not. test_component_integer_length_accepted()) all_passed = .false.
    if (.not. test_declared_character_dummy_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: invalid character length expressions are rejected'

contains

    function declaration_source(spec) result(source)
        character(len=*), intent(in) :: spec
        character(len=:), allocatable :: source

        source = 'program main'//new_line('a')// &
                 '  character('//spec//') :: c = '' '''//new_line('a')// &
                 '  print *, c'//new_line('a')// &
                 'end program main'
    end function declaration_source

    logical function test_real_length_rejected()
        test_real_length_rejected = expect_error_contains( &
            declaration_source('1.'), FRAGMENT, &
            '/tmp/ffc_session_reject_charlen_real')
    end function test_real_length_rejected

    logical function test_double_length_rejected()
        test_double_length_rejected = expect_error_contains( &
            declaration_source('1d1'), FRAGMENT, &
            '/tmp/ffc_session_reject_charlen_double')
    end function test_double_length_rejected

    logical function test_complex_length_rejected()
        test_complex_length_rejected = expect_error_contains( &
            declaration_source('(0.,1.)'), FRAGMENT, &
            '/tmp/ffc_session_reject_charlen_complex')
    end function test_complex_length_rejected

    logical function test_logical_length_rejected()
        test_logical_length_rejected = expect_error_contains( &
            declaration_source('.true.'), FRAGMENT, &
            '/tmp/ffc_session_reject_charlen_logical')
    end function test_logical_length_rejected

    logical function test_character_component_length_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type t'//new_line('a')// &
            '    character(('''''''')) :: c = ''c'''//new_line('a')// &
            '  end type'//new_line('a')// &
            'end program main'

        test_character_component_length_rejected = expect_error_contains( &
            source, FRAGMENT, '/tmp/ffc_session_reject_charlen_component')
    end function test_character_component_length_rejected

    logical function test_implicit_nonconstant_length_rejected()
        ! An IMPLICIT character length must be a constant expression; len(f)
        ! refers to the function result whose own length it would define.
        character(len=*), parameter :: source = &
            'function f(x)'//new_line('a')// &
            'implicit character(len(f)) (x)'//new_line('a')// &
            'character(len(x)) f'//new_line('a')// &
            'end function f'

        test_implicit_nonconstant_length_rejected = expect_error_contains( &
            source, FRAGMENT, '/tmp/ffc_session_reject_charlen_implicit')
    end function test_implicit_nonconstant_length_rejected

    logical function test_integer_length_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(3) :: c = ''abc'''//new_line('a')// &
            '  stop len(c)'//new_line('a')// &
            'end program main'

        test_integer_length_accepted = expect_exit_status( &
            source, 3, '/tmp/ffc_session_charlen_integer_ok')
    end function test_integer_length_accepted

    logical function test_component_integer_length_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type t'//new_line('a')// &
            '    character(2) :: c'//new_line('a')// &
            '  end type'//new_line('a')// &
            '  type(t) :: v'//new_line('a')// &
            '  v%c = ''ab'''//new_line('a')// &
            '  stop len(v%c)'//new_line('a')// &
            'end program main'

        test_component_integer_length_accepted = expect_exit_status( &
            source, 2, '/tmp/ffc_session_charlen_component_ok')
    end function test_component_integer_length_accepted

    logical function test_declared_character_dummy_accepted()
        ! The corrected neighbour of the self-referential IMPLICIT length: the
        ! dummy is declared CHARACTER, so its length is a valid INTEGER value.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(4) :: s'//new_line('a')// &
            '  s = ''abcd'''//new_line('a')// &
            '  stop width(s)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function width(t)'//new_line('a')// &
            '    character(len=*), intent(in) :: t'//new_line('a')// &
            '    width = len(t)'//new_line('a')// &
            '  end function width'//new_line('a')// &
            'end program main'

        test_declared_character_dummy_accepted = expect_exit_status( &
            source, 4, '/tmp/ffc_session_charlen_dummy_ok')
    end function test_declared_character_dummy_accepted

end program test_session_reject_charlen_01_compiler
