program test_session_typed_array_parameter
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== typed scalar and array named constant folding test ==='

    all_passed = .true.
    if (.not. test_typed_scalar_constants()) all_passed = .false.
    if (.not. test_typed_array_constants()) all_passed = .false.
    if (.not. test_complex_array_parameter()) all_passed = .false.
    if (.not. test_array_parameter_in_bounds()) all_passed = .false.
    if (.not. test_nonconstant_rhs_rejected()) all_passed = .false.
    if (.not. test_illegal_conversion_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: typed scalar and array named constants fold at compile time'

contains

    ! Named constants of every intrinsic scalar type keep their declared
    ! kind and print exactly like gfortran.
    logical function test_typed_scalar_constants()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer(8), parameter :: big = 5_8'//new_line('a')// &
            '  real(8), parameter :: r = 2.5d0'//new_line('a')// &
            '  logical, parameter :: flag = .true.'//new_line('a')// &
            '  character(len=3), parameter :: tag = ''abc'''//new_line('a')// &
            '  complex, parameter :: z = (1.0, 2.0)'//new_line('a')// &
            '  print *, big, r, flag, tag, z'//new_line('a')// &
            'end program main'

        test_typed_scalar_constants = expect_output( &
            source, &
            '                    5   2.5000000000000000      T abc'// &
            '             (1.00000000,2.00000000)'//new_line('a'), &
            '/tmp/ffc_typed_param_scalars')
    end function test_typed_scalar_constants

    ! Fixed-size array named constants of each type fold elementwise.
    logical function test_typed_array_constants()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: base(3) = [4, 5, 6]'//new_line('a')// &
            '  real(8), parameter :: rc(2) = [1.0d0, 2.0d0]'//new_line('a')// &
            '  logical, parameter :: lc(2) = [.true., .false.]'//new_line('a')// &
            '  character(len=2), parameter :: cc(2) = [''ab'', ''cd'']'// &
            new_line('a')// &
            '  integer(8), parameter :: ic(2) = [7_8, 8_8]'//new_line('a')// &
            '  print *, base(2), rc(1), lc(1), lc(2), cc(2)'//new_line('a')// &
            '  print *, ic(1) + ic(2)'//new_line('a')// &
            'end program main'

        test_typed_array_constants = expect_output( &
            source, &
            '           5   1.0000000000000000      T F cd'//new_line('a')// &
            '                   15'//new_line('a'), &
            '/tmp/ffc_typed_param_arrays')
    end function test_typed_array_constants

    ! A complex array named constant folds its constructor elements and
    ! stays addressable in executable expressions.
    logical function test_complex_array_parameter()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex, parameter :: zc(2) = [(1.0, 0.0), (0.0, 1.0)]'// &
            new_line('a')// &
            '  complex(8), parameter :: dz(2) = [(1.5d0, 2.5d0), '// &
            '(3.5d0, 4.5d0)]'//new_line('a')// &
            '  print *, zc(1), zc(2)'//new_line('a')// &
            '  print *, dz(2)'//new_line('a')// &
            'end program main'

        test_complex_array_parameter = expect_output( &
            source, &
            '             (1.00000000,0.00000000)'// &
            '             (0.00000000,1.00000000)'//new_line('a')// &
            '               (3.5000000000000000,4.5000000000000000)'// &
            new_line('a'), &
            '/tmp/ffc_typed_param_complex_array')
    end function test_complex_array_parameter

    ! Array named constants are usable as compile-time array bounds.
    logical function test_array_parameter_in_bounds()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: dims(2) = [2, 3]'//new_line('a')// &
            '  integer :: m(dims(1)*dims(2))'//new_line('a')// &
            '  integer :: a(dims(2))'//new_line('a')// &
            '  print *, size(m), size(a)'//new_line('a')// &
            'end program main'

        test_array_parameter_in_bounds = expect_output( &
            source, '           6           3'//new_line('a'), &
            '/tmp/ffc_typed_param_bounds')
    end function test_array_parameter_in_bounds

    ! A nonconstant right-hand side is rejected, not silently folded.
    logical function test_nonconstant_rhs_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  integer, parameter :: n = i'//new_line('a')// &
            '  print *, n'//new_line('a')// &
            'end program main'

        test_nonconstant_rhs_rejected = expect_error_contains( &
            source, 'does not reduce to a constant expression', &
            '/tmp/ffc_typed_param_nonconst')
    end function test_nonconstant_rhs_rejected

    ! A character value initializing an integer named constant is rejected.
    logical function test_illegal_conversion_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: n = ''abc'''//new_line('a')// &
            '  print *, n'//new_line('a')// &
            'end program main'

        test_illegal_conversion_rejected = expect_error_contains( &
            source, 'invalid integer literal', &
            '/tmp/ffc_typed_param_badconv')
    end function test_illegal_conversion_rejected

end program test_session_typed_array_parameter
