program test_session_reject_array_02_compiler
    ! #388: array constructor compatibility.
    !   * (/ ... ] and [ ... /) mix the constructor delimiters,
    !   * a structure constructor cannot initialise an intrinsic type and an
    !     array constructor cannot initialise a scalar,
    !   * a LOGICAL array constructor does not convert to a REAL variable.
    ! Each invalid form gets a source diagnostic; the corrected neighbours
    ! still compile and run.
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    character(len=*), parameter :: DELIM_FRAGMENT = 'array constructor'
    character(len=*), parameter :: TYPE_FRAGMENT = 'cannot convert TYPE('
    character(len=*), parameter :: RANK_FRAGMENT = 'incompatible ranks 0 and 1'
    character(len=*), parameter :: CONVERT_FRAGMENT = 'cannot convert LOGICAL'
    logical :: all_passed

    print *, '=== array constructor compatibility rejection test ==='

    all_passed = .true.
    if (.not. test_paren_open_bracket_close_rejected()) all_passed = .false.
    if (.not. test_bracket_open_paren_close_rejected()) all_passed = .false.
    if (.not. test_structure_ctor_init_rejected()) all_passed = .false.
    if (.not. test_array_ctor_scalar_init_rejected()) all_passed = .false.
    if (.not. test_logical_ctor_to_real_rejected()) all_passed = .false.
    if (.not. test_matched_delimiters_accepted()) all_passed = .false.
    if (.not. test_scalar_char_init_accepted()) all_passed = .false.
    if (.not. test_real_ctor_to_real_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: incompatible array constructors are rejected'

contains

    logical function test_paren_open_bracket_close_rejected()
        ! gfortran.dg/array_constructor_2.f90: a = (/ 1, 2, 3, 4 ]
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            '  a = (/ 1, 2, 3, 4 ]'//new_line('a')// &
            'end program main'

        test_paren_open_bracket_close_rejected = expect_error_contains( &
            source, DELIM_FRAGMENT, '/tmp/ffc_session_reject_array02_delim1')
    end function test_paren_open_bracket_close_rejected

    logical function test_bracket_open_paren_close_rejected()
        ! gfortran.dg/array_constructor_2.f90: a = (/ [ 1, 2, 3, 4 /) ]
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            '  a = (/ [ 1, 2, 3, 4 /) ]'//new_line('a')// &
            'end program main'

        test_bracket_open_paren_close_rejected = expect_error_contains( &
            source, DELIM_FRAGMENT, '/tmp/ffc_session_reject_array02_delim2')
    end function test_bracket_open_paren_close_rejected

    logical function test_structure_ctor_init_rejected()
        ! gfortran.dg/initialization_23.f90: CHARACTER, PARAMETER :: x =
        ! one_parameter('c')
        character(len=*), parameter :: source = &
            'module cdf_aux_mod'//new_line('a')// &
            '  public'//new_line('a')// &
            '  type :: one_parameter'//new_line('a')// &
            '    character :: name'//new_line('a')// &
            '  end type one_parameter'//new_line('a')// &
            '  character, parameter :: the_alpha = one_parameter(''c'')'// &
            new_line('a')// &
            'end module cdf_aux_mod'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use cdf_aux_mod'//new_line('a')// &
            '  print *, the_alpha'//new_line('a')// &
            'end program main'

        test_structure_ctor_init_rejected = expect_error_contains( &
            source, TYPE_FRAGMENT, '/tmp/ffc_session_reject_array02_type')
    end function test_structure_ctor_init_rejected

    logical function test_array_ctor_scalar_init_rejected()
        ! gfortran.dg/initialization_23.f90: CHARACTER, PARAMETER :: x =
        ! (/one_parameter('c')/)
        character(len=*), parameter :: source = &
            'module cdf_aux_mod'//new_line('a')// &
            '  public'//new_line('a')// &
            '  type :: one_parameter'//new_line('a')// &
            '    character :: name'//new_line('a')// &
            '  end type one_parameter'//new_line('a')// &
            '  character, parameter :: the_beta = (/one_parameter(''c'')/)'// &
            new_line('a')// &
            'end module cdf_aux_mod'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use cdf_aux_mod'//new_line('a')// &
            '  print *, the_beta'//new_line('a')// &
            'end program main'

        test_array_ctor_scalar_init_rejected = expect_error_contains( &
            source, RANK_FRAGMENT, '/tmp/ffc_session_reject_array02_rank')
    end function test_array_ctor_scalar_init_rejected

    logical function test_logical_ctor_to_real_rejected()
        ! gfortran.dg/logical_assignment_1.f90: a = [logical::]
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: a(0)'//new_line('a')// &
            '  a = [logical::]'//new_line('a')// &
            '  print *, size(a)'//new_line('a')// &
            'end program main'

        test_logical_ctor_to_real_rejected = expect_error_contains( &
            source, CONVERT_FRAGMENT, '/tmp/ffc_session_reject_array02_convert')
    end function test_logical_ctor_to_real_rejected

    logical function test_matched_delimiters_accepted()
        ! Corrected neighbour: matching delimiters on both constructors.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            '  a = (/ 1, 2, 3, 4 /)'//new_line('a')// &
            '  a = [ 1, 2, 3, 4 ]'//new_line('a')// &
            '  stop a(3)'//new_line('a')// &
            'end program main'

        test_matched_delimiters_accepted = expect_exit_status( &
            source, 3, '/tmp/ffc_session_array02_delim_ok')
    end function test_matched_delimiters_accepted

    logical function test_scalar_char_init_accepted()
        ! Corrected neighbour: a scalar parameter takes a scalar initializer
        ! of its own type.
        character(len=*), parameter :: source = &
            'module cdf_aux_mod'//new_line('a')// &
            '  public'//new_line('a')// &
            '  integer, parameter :: the_alpha = 4'//new_line('a')// &
            'end module cdf_aux_mod'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use cdf_aux_mod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  stop the_alpha'//new_line('a')// &
            'end program main'

        test_scalar_char_init_accepted = expect_exit_status( &
            source, 4, '/tmp/ffc_session_array02_scalar_init_ok')
    end function test_scalar_char_init_accepted

    logical function test_real_ctor_to_real_accepted()
        ! Corrected neighbour: a REAL array constructor assigns to a REAL
        ! array.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: a(2)'//new_line('a')// &
            '  a = [1.0, 2.0]'//new_line('a')// &
            '  stop int(a(2)) + 3'//new_line('a')// &
            'end program main'

        test_real_ctor_to_real_accepted = expect_exit_status( &
            source, 5, '/tmp/ffc_session_array02_real_ctor_ok')
    end function test_real_ctor_to_real_accepted

end program test_session_reject_array_02_compiler
