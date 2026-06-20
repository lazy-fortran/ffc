program test_session_derived_character_component_compiler
    ! Fixed-length character components take a place in the flat slot layout
    ! (ceil(len/4) byte-storage slots) so a derived type that declares them
    ! lowers and runs. Reading or writing a character component through obj%comp
    ! is not yet supported and must report a clear diagnostic rather than emit a
    ! wrong i32 access (#265).
    use ffc_test_support, only: expect_no_error, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session derived character component test ==='

    all_passed = .true.
    if (.not. test_character_component_type_lowers()) all_passed = .false.
    if (.not. test_nested_character_component_lowers()) all_passed = .false.
    if (.not. test_character_component_access_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: derived character components lower into the slot layout'

contains

    logical function test_character_component_type_lowers()
        ! A derived type with mixed numeric and character components lowers and
        ! the program (which never touches the components) runs.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: person_t'//new_line('a')// &
            '    integer :: age'//new_line('a')// &
            '    character(len=20) :: name'//new_line('a')// &
            '    real :: height'//new_line('a')// &
            '  end type person_t'//new_line('a')// &
            '  type(person_t) :: p'//new_line('a')// &
            '  p%age = 7'//new_line('a')// &
            '  print *, p%age'//new_line('a')// &
            'end program main'

        test_character_component_type_lowers = expect_no_error( &
            source, '/tmp/ffc_derived_char_component_test')
    end function test_character_component_type_lowers

    logical function test_nested_character_component_lowers()
        ! A type with a nested derived component that itself carries a character
        ! component lowers (the corpus derived_type_nested case).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: address_t'//new_line('a')// &
            '    integer :: street_num'//new_line('a')// &
            '    character(len=30) :: city'//new_line('a')// &
            '  end type address_t'//new_line('a')// &
            '  type :: person_t'//new_line('a')// &
            '    character(len=20) :: name'//new_line('a')// &
            '    type(address_t) :: address'//new_line('a')// &
            '  end type person_t'//new_line('a')// &
            '  type(person_t) :: p'//new_line('a')// &
            '  print *, "ok"'//new_line('a')// &
            'end program main'

        test_nested_character_component_lowers = expect_no_error( &
            source, '/tmp/ffc_derived_nested_char_component_test')
    end function test_nested_character_component_lowers

    logical function test_character_component_access_rejected()
        ! Writing a character component must report the unsupported diagnostic,
        ! not silently store through an i32 width.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: person_t'//new_line('a')// &
            '    character(len=8) :: name'//new_line('a')// &
            '  end type person_t'//new_line('a')// &
            '  type(person_t) :: p'//new_line('a')// &
            '  p%name = "Ada"'//new_line('a')// &
            '  print *, p%name'//new_line('a')// &
            'end program main'

        test_character_component_access_rejected = expect_error_contains( &
            source, 'character component access', &
            '/tmp/ffc_derived_char_component_reject_test')
    end function test_character_component_access_rejected

end program test_session_derived_character_component_compiler
