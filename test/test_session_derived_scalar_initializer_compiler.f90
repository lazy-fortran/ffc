program test_session_derived_scalar_initializer_compiler
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session derived scalar initializer compiler test ==='

    all_passed = .true.
    if (.not. test_constructor_initializer()) all_passed = .false.
    if (.not. test_constructor_initializer_with_default()) all_passed = .false.
    if (.not. test_local_real_logical_defaults()) all_passed = .false.
    if (.not. test_program_derived_parameter()) all_passed = .false.
    if (.not. test_variable_initializer_still_unsupported()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: derived scalar initializers lower through direct LIRIC'

contains

    logical function test_constructor_initializer()
        ! type(t) :: v = t(...) materialises the constant structure constructor
        ! into the freshly declared instance.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '    real :: r'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: v = t(3, 2.5)'//new_line('a')// &
            '  print *, v%x, v%r'//new_line('a')// &
            'end program main'

        test_constructor_initializer = expect_output( &
            source, '           3   2.50000000    '//new_line('a'), &
            '/tmp/ffc_session_derived_init_ctor_test')
    end function test_constructor_initializer

    logical function test_constructor_initializer_with_default()
        ! A component omitted from the constructor keeps its compile-time default.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '    integer :: y = 9'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: v = t(4)'//new_line('a')// &
            '  print *, v%x, v%y'//new_line('a')// &
            'end program main'

        test_constructor_initializer_with_default = expect_output( &
            source, '           4           9'//new_line('a'), &
            '/tmp/ffc_session_derived_init_default_test')
    end function test_constructor_initializer_with_default

    logical function test_local_real_logical_defaults()
        ! A default-initialised local derived variable reads real (f32) and
        ! logical component defaults.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type t'//new_line('a')// &
            '    logical :: b = .true.'//new_line('a')// &
            '    real :: r = 1.5'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: a'//new_line('a')// &
            '  print *, a%b, a%r'//new_line('a')// &
            'end program main'

        test_local_real_logical_defaults = expect_output( &
            source, ' T   1.50000000    '//new_line('a'), &
            '/tmp/ffc_session_derived_local_default_test')
    end function test_local_real_logical_defaults

    logical function test_variable_initializer_still_unsupported()
        ! A nested-derived constructor initializer stays unsupported and keeps a
        ! clean diagnostic rather than miscompiling.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type inner'//new_line('a')// &
            '    integer :: a'//new_line('a')// &
            '  end type inner'//new_line('a')// &
            '  type outer'//new_line('a')// &
            '    type(inner) :: i'//new_line('a')// &
            '  end type outer'//new_line('a')// &
            '  type(outer) :: v = outer(inner(2))'//new_line('a')// &
            '  print *, v%i%a'//new_line('a')// &
            'end program main'

        test_variable_initializer_still_unsupported = expect_error_contains( &
            source, 'derived type constructor', &
            '/tmp/ffc_session_derived_init_unsupported_test')
    end function test_variable_initializer_still_unsupported

    logical function test_program_derived_parameter()
        ! A program-scope scalar derived parameter with a constructor
        ! initialiser lowers like an initialised local; its components read back.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type t'//new_line('a')// &
            '    integer :: x = 1'//new_line('a')// &
            '    integer :: y = 1'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t), parameter :: p = t(2, 3)'//new_line('a')// &
            '  print *, p%x, p%y'//new_line('a')// &
            'end program main'

        test_program_derived_parameter = expect_output( &
            source, '           2           3'//new_line('a'), &
            '/tmp/ffc_session_derived_prog_param_test')
    end function test_program_derived_parameter

end program test_session_derived_scalar_initializer_compiler
