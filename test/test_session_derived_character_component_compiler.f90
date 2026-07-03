program test_session_derived_character_component_compiler
    ! Fixed-length character components take a place in the flat slot layout
    ! (ceil((len+1)/4) byte-storage slots, the extra byte holding a NUL
    ! terminator) so a derived type that declares them lowers and runs.
    ! Reading and writing a character component through obj%comp, printing it,
    ! and passing a literal to a structure constructor are all supported
    ! (#265); a character ARRAY component stays unsupported.
    use ffc_test_support, only: expect_no_error, expect_error_contains, &
        expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session derived character component test ==='

    all_passed = .true.
    if (.not. test_character_component_type_lowers()) all_passed = .false.
    if (.not. test_nested_character_component_lowers()) all_passed = .false.
    if (.not. test_character_component_read_write()) all_passed = .false.
    if (.not. test_character_component_constructor()) all_passed = .false.
    if (.not. test_character_component_argument()) all_passed = .false.
    if (.not. test_character_array_component_rejected()) all_passed = .false.

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

    logical function test_character_component_read_write()
        ! Writing then reading a fixed-length character component: blank-pad,
        ! truncate on overflow, compare, and print all agree with gfortran.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: person_t'//new_line('a')// &
            '    integer :: age'//new_line('a')// &
            '    character(len=8) :: name'//new_line('a')// &
            '  end type person_t'//new_line('a')// &
            '  type(person_t) :: p'//new_line('a')// &
            '  p%age = 7'//new_line('a')// &
            '  p%name = "Ada"'//new_line('a')// &
            '  if (p%name /= "Ada") error stop'//new_line('a')// &
            '  print *, p%age'//new_line('a')// &
            '  print *, p%name'//new_line('a')// &
            '  p%name = "Grace Hopper"'//new_line('a')// &
            '  print *, p%name'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           7'//new_line('a')// &
            ' Ada     '//new_line('a')// &
            ' Grace Ho'//new_line('a')

        test_character_component_read_write = expect_output( &
            source, expected, '/tmp/ffc_derived_char_component_rw_test')
    end function test_character_component_read_write

    logical function test_character_component_constructor()
        ! A structure constructor's character-literal argument stores into
        ! the component's fixed-length slot, and the field concatenates and
        ! prints back out like any other character value.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: person_t'//new_line('a')// &
            '    character(len=8) :: name'//new_line('a')// &
            '    integer :: age'//new_line('a')// &
            '  end type person_t'//new_line('a')// &
            '  type(person_t) :: p'//new_line('a')// &
            '  p = person_t("Ada", 7)'//new_line('a')// &
            '  print *, p%name, p%age'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            ' Ada                7'//new_line('a')

        test_character_component_constructor = expect_output( &
            source, expected, '/tmp/ffc_derived_char_component_ctor_test')
    end function test_character_component_constructor

    logical function test_character_component_argument()
        ! A character component passed as an actual argument builds the same
        ! {data, length} descriptor as a plain character(len=*) variable.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: person_t'//new_line('a')// &
            '    character(len=8) :: name'//new_line('a')// &
            '  end type person_t'//new_line('a')// &
            '  type(person_t) :: p'//new_line('a')// &
            '  p%name = "Ada"'//new_line('a')// &
            '  call greet(p%name)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine greet(s)'//new_line('a')// &
            '    character(len=*), intent(in) :: s'//new_line('a')// &
            '    print *, "Hello, ", trim(s)'//new_line('a')// &
            '  end subroutine greet'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            ' Hello, Ada'//new_line('a')

        test_character_component_argument = expect_output( &
            source, expected, '/tmp/ffc_derived_char_component_arg_test')
    end function test_character_component_argument

    logical function test_character_array_component_rejected()
        ! A character ARRAY component stays unsupported: the flat scalar slot
        ! layout only reserves inline byte storage for a scalar character.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: roster_t'//new_line('a')// &
            '    character(len=4) :: tags(2)'//new_line('a')// &
            '  end type roster_t'//new_line('a')// &
            '  type(roster_t) :: r'//new_line('a')// &
            '  print *, "unused"'//new_line('a')// &
            'end program main'

        test_character_array_component_rejected = expect_error_contains( &
            source, 'character array', &
            '/tmp/ffc_derived_char_array_component_reject_test')
    end function test_character_array_component_rejected

end program test_session_derived_character_component_compiler
