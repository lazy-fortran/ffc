program test_session_reject_text_helpers
    use session_program_lowering_reject_text, only: &
        normalized_base_type, base_type_root, implicit_base_type, &
        starts_with_word
    implicit none

    logical :: all_passed

    all_passed = test_normalized_types() .and. test_type_roots() .and. &
        test_implicit_types() .and. test_word_boundaries()
    if (.not. all_passed) stop 1
    print *, 'PASS: rejection text helpers'

contains

    logical function test_normalized_types()
        test_normalized_types = normalized_base_type(' CLASS( REAL(KIND=8) ) ') == &
            'type(real(kind=8))' .and. &
            normalized_base_type(' type ( integer ) ') == 'type(integer)' .and. &
            normalized_base_type(' class(*) ') == 'class(*)'
    end function test_normalized_types

    logical function test_type_roots()
        test_type_roots = base_type_root('real(kind=8)') == 'real' .and. &
            base_type_root('character(len=12)') == 'character' .and. &
            base_type_root('type(point)') == 'type(point)' .and. &
            base_type_root('class(*)') == 'class(*)'
    end function test_type_roots

    logical function test_implicit_types()
        test_implicit_types = implicit_base_type('index') == 'integer' .and. &
            implicit_base_type('name') == 'integer' .and. &
            implicit_base_type('alpha') == 'real' .and. &
            implicit_base_type('') == ''
    end function test_implicit_types

    logical function test_word_boundaries()
        test_word_boundaries = starts_with_word('character(len=4)', 'character') .and. &
            starts_with_word('logical', 'logical') .and. &
            .not. starts_with_word('characteristic', 'character') .and. &
            .not. starts_with_word('real', 'logical')
    end function test_word_boundaries

end program test_session_reject_text_helpers
