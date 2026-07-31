program test_session_reject_boz_01_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== BOZ literal context rejection compiler test ==='

    all_passed = .true.
    if (.not. test_boz_to_class_star_rejected()) all_passed = .false.
    if (.not. test_boz_associate_selector_rejected()) all_passed = .false.
    if (.not. test_boz_structure_constructor_rejected()) all_passed = .false.
    if (.not. test_boz_integer_assignment_accepted()) all_passed = .false.
    if (.not. test_associate_integer_selector_accepted()) all_passed = .false.
    if (.not. test_data_integer_component_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: BOZ literals rejected outside permitted contexts'

contains

    logical function test_boz_to_class_star_rejected()
        !! gfortran.dg/pr93601.f90: a BOZ literal may only be assigned to an
        !! integer or real variable; an unlimited polymorphic target has no
        !! such type to take the bit pattern.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '   class(*), allocatable :: z'//new_line('a')// &
            "   z = z'1'"//new_line('a')// &
            'end'

        test_boz_to_class_star_rejected = expect_error_contains( &
            source, 'BOZ literal constant', &
            '/tmp/ffc_session_boz01_class_reject')
    end function test_boz_to_class_star_rejected

    logical function test_boz_associate_selector_rejected()
        !! gfortran.dg/pr93603.f90: an associate selector cannot be a BOZ
        !! literal constant - it has no type to associate with.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            "  associate (y => z'1')"//new_line('a')// &
            '  end associate'//new_line('a')// &
            'end'

        test_boz_associate_selector_rejected = expect_error_contains( &
            source, 'BOZ literal constant', &
            '/tmp/ffc_session_boz01_assoc_reject')
    end function test_boz_associate_selector_rejected

    logical function test_boz_structure_constructor_rejected()
        !! gfortran.dg/pr93604.f90: a BOZ literal is not a valid structure
        !! constructor component value, not even in a DATA statement.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '   type t'//new_line('a')// &
            '      integer :: a'//new_line('a')// &
            '   end type'//new_line('a')// &
            '   type(t) :: x'//new_line('a')// &
            "   data x /t(z'1')/"//new_line('a')// &
            'end'

        test_boz_structure_constructor_rejected = expect_error_contains( &
            source, 'BOZ literal constant', &
            '/tmp/ffc_session_boz01_ctor_reject')
    end function test_boz_structure_constructor_rejected

    logical function test_boz_integer_assignment_accepted()
        !! Corrected neighbour of pr93601: an integer target still accepts a
        !! BOZ literal.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '   integer :: z'//new_line('a')// &
            "   z = z'1'"//new_line('a')// &
            '   print *, z'//new_line('a')// &
            '   stop 0'//new_line('a')// &
            'end'

        test_boz_integer_assignment_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_boz01_class_accept')
    end function test_boz_integer_assignment_accepted

    logical function test_associate_integer_selector_accepted()
        !! Corrected neighbour of pr93603: a typed selector associates fine.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            "  associate (y => int(z'1'))"//new_line('a')// &
            '    print *, y'//new_line('a')// &
            '  end associate'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end'

        test_associate_integer_selector_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_boz01_assoc_accept')
    end function test_associate_integer_selector_accepted

    logical function test_data_integer_component_accepted()
        !! Corrected neighbour of pr93604: an ordinary integer constant in the
        !! structure constructor keeps the DATA statement valid.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '   type t'//new_line('a')// &
            '      integer :: a'//new_line('a')// &
            '   end type'//new_line('a')// &
            '   type(t) :: x'//new_line('a')// &
            '   data x /t(1)/'//new_line('a')// &
            '   print *, x%a'//new_line('a')// &
            '   stop 0'//new_line('a')// &
            'end'

        test_data_integer_component_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_boz01_ctor_accept')
    end function test_data_integer_component_accepted

end program test_session_reject_boz_01_compiler
