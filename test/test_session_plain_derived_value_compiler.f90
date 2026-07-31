program test_session_plain_derived_value_compiler
    use ffc_test_support, only: expect_exit_status, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session plain derived value compiler test ==='

    all_passed = .true.
    if (.not. test_nested_constructor_update_copy()) all_passed = .false.
    if (.not. test_same_named_components_keep_layouts()) all_passed = .false.
    if (.not. test_missing_constructor_component_diagnostic()) all_passed = .false.
    if (.not. test_duplicate_constructor_component_diagnostic()) all_passed = .false.
    if (.not. test_incompatible_nested_constructor_diagnostic()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: plain derived values lower through direct LIRIC'

contains

    logical function test_nested_constructor_update_copy()
        ! Construct a nested plain scalar, update its leaf, then copy the whole
        ! value. The result proves nested offsets and whole-value assignment agree.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: leaf_t'//new_line('a')// &
            '    integer :: value'//new_line('a')// &
            '  end type leaf_t'//new_line('a')// &
            '  type :: outer_t'//new_line('a')// &
            '    type(leaf_t) :: leaf'//new_line('a')// &
            '    integer :: tag'//new_line('a')// &
            '  end type outer_t'//new_line('a')// &
            '  type(outer_t) :: original, copy'//new_line('a')// &
            '  original = outer_t(leaf_t(4), 6)'//new_line('a')// &
            '  original%leaf%value = original%leaf%value + 2'//new_line('a')// &
            '  copy = original'//new_line('a')// &
            '  stop copy%leaf%value + copy%tag'//new_line('a')// &
            'end program main'

        ! 4 + 2 + 6 = 12.
        test_nested_constructor_update_copy = expect_exit_status( &
            source, 12, '/tmp/ffc_session_plain_derived_nested')
    end function test_nested_constructor_update_copy

    logical function test_same_named_components_keep_layouts()
        ! Both types have a component named value, but their declared layouts
        ! differ. Resolution must use the containing type, not the text name.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: short_t'//new_line('a')// &
            '    integer :: value'//new_line('a')// &
            '  end type short_t'//new_line('a')// &
            '  type :: long_t'//new_line('a')// &
            '    integer :: value'//new_line('a')// &
            '    integer :: extra'//new_line('a')// &
            '  end type long_t'//new_line('a')// &
            '  type(short_t) :: short_value'//new_line('a')// &
            '  type(long_t) :: long_value'//new_line('a')// &
            '  short_value = short_t(3)'//new_line('a')// &
            '  long_value = long_t(4, 9)'//new_line('a')// &
            '  stop short_value%value + long_value%value + long_value%extra'// &
            new_line('a')// &
            'end program main'

        test_same_named_components_keep_layouts = expect_exit_status( &
            source, 16, '/tmp/ffc_session_plain_derived_same_names')
    end function test_same_named_components_keep_layouts

    logical function test_missing_constructor_component_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: pair_t'//new_line('a')// &
            '    integer :: left'//new_line('a')// &
            '    integer :: right'//new_line('a')// &
            '  end type pair_t'//new_line('a')// &
            '  type(pair_t) :: pair'//new_line('a')// &
            '  pair = pair_t(1)'//new_line('a')// &
            'end program main'

        test_missing_constructor_component_diagnostic = expect_error_contains( &
            source, 'unsupported derived type constructor', &
            '/tmp/ffc_session_plain_derived_missing')
    end function test_missing_constructor_component_diagnostic

    logical function test_duplicate_constructor_component_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: pair_t'//new_line('a')// &
            '    integer :: left'//new_line('a')// &
            '    integer :: right'//new_line('a')// &
            '  end type pair_t'//new_line('a')// &
            '  type(pair_t) :: pair'//new_line('a')// &
            '  pair = pair_t(left=1, left=2)'//new_line('a')// &
            'end program main'

        test_duplicate_constructor_component_diagnostic = expect_error_contains( &
            source, 'unsupported derived type constructor', &
            '/tmp/ffc_session_plain_derived_duplicate')
    end function test_duplicate_constructor_component_diagnostic

    logical function test_incompatible_nested_constructor_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: expected_t'//new_line('a')// &
            '    integer :: value'//new_line('a')// &
            '  end type expected_t'//new_line('a')// &
            '  type :: other_t'//new_line('a')// &
            '    integer :: value'//new_line('a')// &
            '  end type other_t'//new_line('a')// &
            '  type :: wrapper_t'//new_line('a')// &
            '    type(expected_t) :: item'//new_line('a')// &
            '  end type wrapper_t'//new_line('a')// &
            '  type(wrapper_t) :: wrapper'//new_line('a')// &
            '  wrapper = wrapper_t(other_t(3))'//new_line('a')// &
            'end program main'

        test_incompatible_nested_constructor_diagnostic = expect_error_contains( &
            source, 'derived type constructor', &
            '/tmp/ffc_session_plain_derived_incompatible')
    end function test_incompatible_nested_constructor_diagnostic

end program test_session_plain_derived_value_compiler
