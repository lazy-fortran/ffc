program test_session_inquiry_fold
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session kind/size/len constant-folding test ==='

    all_passed = .true.
    if (.not. test_kind_of_double_literal()) all_passed = .false.
    if (.not. test_kind_of_default_literals()) all_passed = .false.
    if (.not. test_kind_of_variable()) all_passed = .false.
    if (.not. test_size_whole_array()) all_passed = .false.
    if (.not. test_size_with_dim()) all_passed = .false.
    if (.not. test_size_as_array_bound()) all_passed = .false.
    if (.not. test_len_fixed_character()) all_passed = .false.
    if (.not. test_nondefault_int_parameter_bound()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: kind/size/len fold to compile-time constants'

contains

    ! kind(1.0d0) folds to 8; the classic dp = kind(1.0d0) idiom.
    logical function test_kind_of_double_literal()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: dp = '// &
            'kind(1.0d0)'//new_line('a')// &
            '  print *, dp'//new_line('a')// &
            'end program main'

        test_kind_of_double_literal = expect_output( &
            source, '           8'//new_line('a'), &
            '/tmp/ffc_session_kind_double')
    end function test_kind_of_double_literal

    ! Default real/integer/logical literals fold to kind 4; character kind 1.
    logical function test_kind_of_default_literals()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: kr = kind(1.0)'// &
            new_line('a')// &
            '  integer, parameter :: ki = kind(1)'// &
            new_line('a')// &
            '  integer, parameter :: kl = kind(.true.)'// &
            new_line('a')// &
            "  integer, parameter :: kc = kind('a')"// &
            new_line('a')// &
            '  print *, kr, ki, kl, kc'//new_line('a')// &
            'end program main'

        test_kind_of_default_literals = expect_output( &
            source, &
            '           4           4           4           1'//new_line('a'), &
            '/tmp/ffc_session_kind_default')
    end function test_kind_of_default_literals

    ! kind(x) of a declared double variable folds to its declared kind.
    logical function test_kind_of_variable()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real(8) :: x'//new_line('a')// &
            '  integer, parameter :: k = kind(x)'// &
            new_line('a')// &
            '  print *, k'//new_line('a')// &
            'end program main'

        test_kind_of_variable = expect_output( &
            source, '           8'//new_line('a'), &
            '/tmp/ffc_session_kind_var')
    end function test_kind_of_variable

    ! size(arr) folds to the total element count of a fixed-size array.
    logical function test_size_whole_array()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3, 4)'//new_line('a')// &
            '  integer, parameter :: n = size(a)'// &
            new_line('a')// &
            '  print *, n'//new_line('a')// &
            'end program main'

        test_size_whole_array = expect_output( &
            source, '          12'//new_line('a'), &
            '/tmp/ffc_session_size_whole')
    end function test_size_whole_array

    ! size(arr, dim) folds to the extent of one dimension.
    logical function test_size_with_dim()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(3, 4)'//new_line('a')// &
            '  integer, parameter :: n1 = size(a, 1)'// &
            new_line('a')// &
            '  integer, parameter :: n2 = size(a, 2)'// &
            new_line('a')// &
            '  print *, n1, n2'//new_line('a')// &
            'end program main'

        test_size_with_dim = expect_output( &
            source, '           3           4'//new_line('a'), &
            '/tmp/ffc_session_size_dim')
    end function test_size_with_dim

    ! A folded size is usable as an array bound.
    logical function test_size_as_array_bound()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5)'//new_line('a')// &
            '  integer :: b(size(a))'//new_line('a')// &
            '  b = 7'//new_line('a')// &
            '  print *, size(b), b(5)'//new_line('a')// &
            'end program main'

        test_size_as_array_bound = expect_output( &
            source, '           5           7'//new_line('a'), &
            '/tmp/ffc_session_size_bound')
    end function test_size_as_array_bound

    ! len(s) folds to the declared length of a fixed-length character.
    logical function test_len_fixed_character()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: s'//new_line('a')// &
            '  integer, parameter :: l = len(s)'// &
            new_line('a')// &
            '  print *, l'//new_line('a')// &
            'end program main'

        test_len_fixed_character = expect_output( &
            source, '           5'//new_line('a'), &
            '/tmp/ffc_session_len_fixed')
    end function test_len_fixed_character

    ! A non-default-kind integer parameter is a compile-time array bound.
    logical function test_nondefault_int_parameter_bound()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use iso_fortran_env, only: int64'// &
            new_line('a')// &
            '  integer(int64), parameter :: n = 5'// &
            new_line('a')// &
            '  integer :: a(n)'//new_line('a')// &
            '  a = 1'//new_line('a')// &
            '  print *, size(a)'//new_line('a')// &
            'end program main'

        test_nondefault_int_parameter_bound = expect_output( &
            source, '           5'//new_line('a'), &
            '/tmp/ffc_session_nondefault_int_param')
    end function test_nondefault_int_parameter_bound

end program test_session_inquiry_fold
