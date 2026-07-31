program test_session_reject_data_01_compiler
    ! DATA object and initializer restrictions (#383). A data-stmt-object must
    ! be a definable variable that is not a named constant, not a pointer and
    ! not already initialized in its declaration; a data-stmt-value must be a
    ! constant expression of the object's type class. Each invalid form is
    ! rejected with its own diagnostic while the corrected neighbour compiles
    ! and runs.
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== DATA object/initializer restriction compiler test ==='

    all_passed = .true.
    if (.not. test_parameter_object_rejected()) all_passed = .false.
    if (.not. test_pointer_object_rejected()) all_passed = .false.
    if (.not. test_initialized_object_rejected()) all_passed = .false.
    if (.not. test_variable_value_rejected()) all_passed = .false.
    if (.not. test_type_mismatch_rejected()) all_passed = .false.
    if (.not. test_nonconstant_initializer_rejected()) all_passed = .false.
    if (.not. test_variable_subscript_value_rejected()) all_passed = .false.
    if (.not. test_old_style_init_rejected()) all_passed = .false.
    if (.not. test_valid_data_statement_accepted()) all_passed = .false.
    if (.not. test_valid_section_data_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: invalid DATA objects and initializers rejected, ' // &
        'valid DATA statements still lowered'

contains

    logical function test_parameter_object_rejected()
        ! A named constant has no storage to initialize (gfortran: "shall not
        ! appear in a DATA statement").
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: a(2) = 1'//new_line('a')// &
            '  data a(2) /a(1)/'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            'end program main'

        test_parameter_object_rejected = expect_error_contains( &
            source, 'shall not appear in a DATA statement', &
            '/tmp/ffc_data01_param_object')
    end function test_parameter_object_rejected

    logical function test_pointer_object_rejected()
        ! A pointer data-stmt-object initializes the target, not the pointer.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, pointer :: x'//new_line('a')// &
            '  data x /2/'//new_line('a')// &
            '  print *, 1'//new_line('a')// &
            'end program main'

        test_pointer_object_rejected = expect_error_contains( &
            source, 'POINTER attribute', '/tmp/ffc_data01_pointer_object')
    end function test_pointer_object_rejected

    logical function test_initialized_object_rejected()
        ! An entity initialized in its declaration cannot be initialized again.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x(2) = 1.0'//new_line('a')// &
            '  data x /1.0, 2.0/'//new_line('a')// &
            '  print *, x'//new_line('a')// &
            'end program main'

        test_initialized_object_rejected = expect_error_contains( &
            source, 'already is initialized', '/tmp/ffc_data01_initialized')
    end function test_initialized_object_rejected

    logical function test_variable_value_rejected()
        ! A data-stmt-value must be constant, so a plain variable is invalid.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i = 0'//new_line('a')// &
            '  integer :: z(2)'//new_line('a')// &
            '  data z /2*i/'//new_line('a')// &
            '  print *, z'//new_line('a')// &
            'end program main'

        test_variable_value_rejected = expect_error_contains( &
            source, 'must be a PARAMETER in DATA statement', &
            '/tmp/ffc_data01_variable_value')
    end function test_variable_value_rejected

    logical function test_type_mismatch_rejected()
        ! A character constant cannot initialize an integer object.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=3), parameter :: mychar(3) = ' // &
            '[ "abc", "def", "ghi" ]'//new_line('a')// &
            '  integer :: c(2)'//new_line('a')// &
            '  data c / mychar(1), mychar(3) /'//new_line('a')// &
            '  print *, c'//new_line('a')// &
            'end program main'

        test_type_mismatch_rejected = expect_error_contains( &
            source, 'incompatible types in DATA statement', &
            '/tmp/ffc_data01_type_mismatch')
    end function test_type_mismatch_rejected

    logical function test_nonconstant_initializer_rejected()
        ! A parameter array indexed by an array element is not a constant
        ! expression; the parser cannot build the statement at all, so the
        ! source form itself is rejected instead of being dropped silently and
        ! leaving the object uninitialised.
        character(len=*), parameter :: source = &
            'integer, parameter, dimension(4) :: myint = [4,3,2,1]'// &
            new_line('a')// &
            'integer :: a(5)'//new_line('a')// &
            'data a(1:2) / myint(a(1)), myint(2) /'//new_line('a')// &
            'end'

        test_nonconstant_initializer_rejected = expect_error_contains( &
            source, 'invalid initializer in DATA statement', &
            '/tmp/ffc_data01_invalid_initializer')
    end function test_nonconstant_initializer_rejected

    logical function test_variable_subscript_value_rejected()
        ! A parameter array indexed by a variable is not a constant expression.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: myint(4) = [4, 3, 2, 1]'//new_line('a')// &
            '  integer :: a(5), b'//new_line('a')// &
            '  data a(1:2) / myint(b), myint(2) /'//new_line('a')// &
            '  print *, a(1)'//new_line('a')// &
            'end program main'

        test_variable_subscript_value_rejected = expect_error_contains( &
            source, 'must be a PARAMETER in DATA statement', &
            '/tmp/ffc_data01_variable_subscript')
    end function test_variable_subscript_value_rejected

    logical function test_old_style_init_rejected()
        ! Old-style slashed initialization gives the entity the DATA attribute.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer z /10/'//new_line('a')// &
            '  print *, z'//new_line('a')// &
            'end program main'

        test_old_style_init_rejected = expect_error_contains( &
            source, 'old-style slashed initialization', &
            '/tmp/ffc_data01_old_style')
    end function test_old_style_init_rejected

    logical function test_valid_data_statement_accepted()
        ! Corrected neighbour: definable objects, constant values of the
        ! object's type class.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: k = 7'//new_line('a')// &
            '  integer :: z(2)'//new_line('a')// &
            '  data z /2*k/'//new_line('a')// &
            '  if (z(1) + z(2) /= 14) stop 1'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_valid_data_statement_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_data01_valid_data')
    end function test_valid_data_statement_accepted

    logical function test_valid_section_data_accepted()
        ! Corrected neighbour of the invalid-initializer case: parameter
        ! array elements with constant subscripts as the values.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: myint(4) = [4, 3, 2, 1]'//new_line('a')// &
            '  integer :: a(2)'//new_line('a')// &
            '  data a / myint(1), myint(2) /'//new_line('a')// &
            '  if (a(1) + a(2) /= 7) stop 1'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_valid_section_data_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_data01_valid_section')
    end function test_valid_section_data_accepted

end program test_session_reject_data_01_compiler
