program test_session_derived_typed_defaults_compiler
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session typed derived component defaults test ==='

    all_passed = .true.
    if (.not. test_default_initialization()) all_passed = .false.
    if (.not. test_constructor_overrides_one_component()) all_passed = .false.
    if (.not. test_constructor_all_defaults()) all_passed = .false.
    if (.not. test_c_null_ptr_default()) all_passed = .false.
    if (.not. test_nonconstant_default_rejected()) all_passed = .false.
    if (.not. test_incompatible_c_ptr_default_rejected()) all_passed = .false.
    if (.not. test_incompatible_character_default_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: typed derived component defaults lower through direct LIRIC'

contains

    function typed_default_type() result(source)
        ! A type whose components carry integer, real(real64), logical, and
        ! fixed-length character defaults. Output below matches gfortran.
        character(len=:), allocatable :: source

        source = &
            '  type :: cfg_t'//new_line('a')// &
            '    integer :: n = 7'//new_line('a')// &
            '    real(real64) :: scale = 2.5_real64'//new_line('a')// &
            '    logical :: on = .true.'//new_line('a')// &
            '    character(len=6) :: tag = "abcde"'//new_line('a')// &
            '  end type cfg_t'//new_line('a')
    end function typed_default_type

    logical function test_default_initialization()
        ! A plain declaration takes every typed component default.
        character(len=:), allocatable :: source

        source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_fortran_env, only: real64'//new_line('a')// &
            typed_default_type()// &
            '  type(cfg_t) :: a'//new_line('a')// &
            '  print *, a%n, a%scale, a%on, a%tag'//new_line('a')// &
            'end program main'

        test_default_initialization = expect_output( &
            source, '           7   2.5000000000000000      T abcde '// &
            new_line('a'), '/tmp/ffc_derived_typed_defaults_init')
    end function test_default_initialization

    logical function test_constructor_overrides_one_component()
        ! cfg_t(3) supplies the first component; the remaining real(real64),
        ! logical, and character components keep their declared defaults.
        character(len=:), allocatable :: source

        source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_fortran_env, only: real64'//new_line('a')// &
            typed_default_type()// &
            '  type(cfg_t) :: b'//new_line('a')// &
            '  b = cfg_t(3)'//new_line('a')// &
            '  print *, b%n, b%scale, b%on, b%tag'//new_line('a')// &
            'end program main'

        test_constructor_overrides_one_component = expect_output( &
            source, '           3   2.5000000000000000      T abcde '// &
            new_line('a'), '/tmp/ffc_derived_typed_defaults_override')
    end function test_constructor_overrides_one_component

    logical function test_constructor_all_defaults()
        ! Every component supplied explicitly still lowers, so the default path
        ! does not shadow an actual argument.
        character(len=:), allocatable :: source

        source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_fortran_env, only: real64'//new_line('a')// &
            typed_default_type()// &
            '  type(cfg_t) :: b'//new_line('a')// &
            '  b = cfg_t(3, 9.25_real64, .false., "zz")'//new_line('a')// &
            '  print *, b%n, b%scale, b%on, b%tag'//new_line('a')// &
            'end program main'

        test_constructor_all_defaults = expect_output( &
            source, '           3   9.2500000000000000      F zz    '// &
            new_line('a'), '/tmp/ffc_derived_typed_defaults_explicit')
    end function test_constructor_all_defaults

    logical function test_c_null_ptr_default()
        ! A type(c_ptr) component may default to c_null_ptr; the neighbouring
        ! component's default must still land at the right slot.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr'// &
            new_line('a')// &
            '  type :: h_t'//new_line('a')// &
            '    type(c_ptr) :: p = c_null_ptr'//new_line('a')// &
            '    integer :: n = 5'//new_line('a')// &
            '  end type h_t'//new_line('a')// &
            '  type(h_t) :: h'//new_line('a')// &
            '  print *, h%n'//new_line('a')// &
            'end program main'

        test_c_null_ptr_default = expect_output( &
            source, '           5'//new_line('a'), &
            '/tmp/ffc_derived_typed_defaults_cptr')
    end function test_c_null_ptr_default

    logical function test_nonconstant_default_rejected()
        ! A default built from a runtime variable is not a constant expression.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_fortran_env, only: real64'//new_line('a')// &
            '  integer :: k'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    real(real64) :: x = real(k, real64)'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: v'//new_line('a')// &
            '  k = 1'//new_line('a')// &
            '  print *, v%x'//new_line('a')// &
            'end program main'

        test_nonconstant_default_rejected = expect_error_contains( &
            source, 'derived type component initializer', &
            '/tmp/ffc_derived_typed_defaults_nonconst')
    end function test_nonconstant_default_rejected

    logical function test_incompatible_c_ptr_default_rejected()
        ! An integer default is not assignable to a type(c_ptr) component.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_c_binding, only: c_ptr'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    type(c_ptr) :: p = 1'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: v'//new_line('a')// &
            '  print *, 1'//new_line('a')// &
            'end program main'

        test_incompatible_c_ptr_default_rejected = expect_error_contains( &
            source, 'derived type component initializer', &
            '/tmp/ffc_derived_typed_defaults_cptr_bad')
    end function test_incompatible_c_ptr_default_rejected

    logical function test_incompatible_character_default_rejected()
        ! An integer default is not assignable to a character component.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    character(len=4) :: s = 3'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: v'//new_line('a')// &
            '  print *, v%s'//new_line('a')// &
            'end program main'

        test_incompatible_character_default_rejected = expect_error_contains( &
            source, 'derived type component initializer', &
            '/tmp/ffc_derived_typed_defaults_char_bad')
    end function test_incompatible_character_default_rejected

end program test_session_derived_typed_defaults_compiler
