program test_session_array_mask_reduction_compiler
    use ffc_test_support, only: expect_error_contains, expect_output, &
        expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    print *, '=== direct session array mask reduction test ==='

    all_passed = .true.
    if (.not. test_array_vs_scalar()) all_passed = .false.
    if (.not. test_array_vs_array()) all_passed = .false.
    if (.not. test_array_vs_constructor()) all_passed = .false.
    if (.not. test_logical_constructor_equivalence()) all_passed = .false.
    if (.not. test_real_and_allocatable()) all_passed = .false.
    if (.not. test_elemental_abs_mask()) all_passed = .false.
    if (.not. test_complex_allocatable_abs_mask()) all_passed = .false.
    if (.not. test_user_elemental_abs_mask()) all_passed = .false.
    if (.not. test_runtime_comparison_mask()) all_passed = .false.
    if (.not. test_runtime_rank2_comparison_mask()) all_passed = .false.
    if (.not. test_runtime_rank3_comparison_mask()) all_passed = .false.
    if (.not. test_runtime_rank4_comparison_mask()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: any/all over whole-array comparisons lower correctly'

contains

    logical function test_array_vs_scalar()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '    integer :: a(3)'//new_line('a')// &
            '    a = [1, 1, 1]'//new_line('a')// &
            '    if (any(a /= 1)) error stop'//new_line('a')// &
            '    if (.not. all(a == 1)) error stop'//new_line('a')// &
            '    print *, "ok"'//new_line('a')// &
            'end program main'

        test_array_vs_scalar = expect_output( &
            source, ' ok'//new_line('a'), '/tmp/ffc_mask_scalar')
    end function test_array_vs_scalar

    logical function test_array_vs_array()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '    integer :: a(3), b(3)'//new_line('a')// &
            '    a = [1, 2, 3]'//new_line('a')// &
            '    b = [1, 2, 3]'//new_line('a')// &
            '    if (any(a /= b)) error stop'//new_line('a')// &
            '    b = [1, 9, 3]'//new_line('a')// &
            '    if (all(a == b)) error stop'//new_line('a')// &
            '    print *, "ok"'//new_line('a')// &
            'end program main'

        test_array_vs_array = expect_output( &
            source, ' ok'//new_line('a'), '/tmp/ffc_mask_array')
    end function test_array_vs_array

    logical function test_array_vs_constructor()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '    integer :: a(4)'//new_line('a')// &
            '    a = [1, 2, 3, 4]'//new_line('a')// &
            '    if (any(a /= [1, 2, 3, 4])) error stop'//new_line('a')// &
            '    print *, "ok"'//new_line('a')// &
            'end program main'

        test_array_vs_constructor = expect_output( &
            source, ' ok'//new_line('a'), '/tmp/ffc_mask_ctor')
    end function test_array_vs_constructor

    logical function test_logical_constructor_equivalence()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '    logical, parameter :: x(2) = [.true., .false.]'//new_line('a')// &
            '    if (all((x .neqv. .true.) .neqv. [.false., .true.])) '// &
            'error stop 1'//new_line('a')// &
            '    if (all(([.false., .true.] .eqv. [.false., .true.]) '// &
            '.neqv. [.true., .true.])) error stop 2'//new_line('a')// &
            '    print *, "ok"'//new_line('a')// &
            'end program main'

        test_logical_constructor_equivalence = expect_output_matches_gfortran( &
            source, 'logical_constructor_equivalence')
    end function test_logical_constructor_equivalence

    logical function test_real_and_allocatable()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '    real :: x(3)'//new_line('a')// &
            '    integer, allocatable :: k(:)'//new_line('a')// &
            '    x = [1.0, 2.0, 3.0]'//new_line('a')// &
            '    if (any(x > 3.5)) error stop'//new_line('a')// &
            '    allocate(k(3))'//new_line('a')// &
            '    k = 5'//new_line('a')// &
            '    if (any(k /= 5)) error stop'//new_line('a')// &
            '    print *, "ok"'//new_line('a')// &
            'end program main'

        test_real_and_allocatable = expect_output( &
            source, ' ok'//new_line('a'), '/tmp/ffc_mask_real_alloc')
    end function test_real_and_allocatable

    logical function test_elemental_abs_mask()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '    real :: samples(5)'//new_line('a')// &
            '    real(8) :: wide(2)'//new_line('a')// &
            '    real :: tolerance'//new_line('a')// &
            '    samples = [1.0, -2.0, 3.0, -4.0, 5.0]'//new_line('a')// &
            '    wide = [-1.0d0, 2.0d0]'//new_line('a')// &
            '    tolerance = 0.1'//new_line('a')// &
            '    if (.not. any(abs(wide) > 1.5d0)) error stop'//new_line('a')// &
            '    if (any(abs(samples) &'//new_line('a')// &
            '        > tolerance)) then'//new_line('a')// &
            '        print *, "values exceed tolerance"'//new_line('a')// &
            '    end if'//new_line('a')// &
            'end program main'

        test_elemental_abs_mask = expect_output( &
            source, ' values exceed tolerance'//new_line('a'), &
            '/tmp/ffc_mask_elemental_abs')
    end function test_elemental_abs_mask

    logical function test_complex_allocatable_abs_mask()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '    complex(8), allocatable :: z(:)'//new_line('a')// &
            '    real(8) :: tolerance, magnitude'//new_line('a')// &
            '    allocate(z(3))'//new_line('a')// &
            '    z = [(1.0d0, 2.0d0), (3.0d0, 4.0d0), (5.0d0, 6.0d0)]'//new_line('a')// &
            '    tolerance = 1.0d-6'//new_line('a')// &
            '    if (.not. any(aimag(z) > tolerance * abs(abs(z)))) error stop 1'//new_line('a')// &
            '    magnitude = abs(z(1))'//new_line('a')// &
            '    if (abs(magnitude - sqrt(5.0d0)) > tolerance) error stop 2'//new_line('a')// &
            '    if (any(abs(abs(z)) < 0.0d0)) error stop 3'//new_line('a')// &
            '    print *, "ok"'//new_line('a')// &
            'end program main'

        test_complex_allocatable_abs_mask = expect_output( &
            source, ' ok'//new_line('a'), '/tmp/ffc_mask_complex_alloc')
    end function test_complex_allocatable_abs_mask

    logical function test_user_elemental_abs_mask()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '    real :: samples(2)'//new_line('a')// &
            '    samples = [-1.0, 2.0]'//new_line('a')// &
            '    if (any(abs(samples) > 0.0)) error stop'//new_line('a')// &
            '    print *, "ok"'//new_line('a')// &
            'contains'//new_line('a')// &
            '    elemental real function abs(x)'//new_line('a')// &
            '        real, intent(in) :: x'//new_line('a')// &
            '        abs = -100.0 + 0.0*x'//new_line('a')// &
            '    end function abs'//new_line('a')// &
            'end program main'

        test_user_elemental_abs_mask = expect_error_contains( &
            source, 'AST node is not an identifier', '/tmp/ffc_mask_user_abs')
    end function test_user_elemental_abs_mask

    logical function test_runtime_comparison_mask()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '    integer, allocatable :: a(:)'//new_line('a')// &
            '    allocate(a(4))'//new_line('a')// &
            '    a = [1, 3, 5, 2]'//new_line('a')// &
            '    if (count(a > 2) /= 2) error stop 1'//new_line('a')// &
            '    if (.not. any(a == 5)) error stop 2'//new_line('a')// &
            '    if (all(a > 2)) error stop 3'//new_line('a')// &
            '    print *, "ok"'//new_line('a')// &
            'end program main'

        test_runtime_comparison_mask = expect_output( &
            source, ' ok'//new_line('a'), '/tmp/ffc_mask_runtime_comparison')
    end function test_runtime_comparison_mask

    logical function test_runtime_rank2_comparison_mask()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '    integer, allocatable :: a(:,:), empty(:,:)'//new_line('a')// &
            '    allocate(a(2,3), empty(0,3))'//new_line('a')// &
            '    a(1,1) = 1; a(2,1) = 3; a(1,2) = 5'//new_line('a')// &
            '    a(2,2) = 2; a(1,3) = 4; a(2,3) = 6'//new_line('a')// &
            '    if (count(a > 2) /= 4) error stop 1'//new_line('a')// &
            '    if (.not. any(a == 5)) error stop 2'//new_line('a')// &
            '    if (.not. all(a > 0)) error stop 3'//new_line('a')// &
            '    if (count(empty > 2) /= 0) error stop 4'//new_line('a')// &
            '    if (any(empty > 2)) error stop 5'//new_line('a')// &
            '    if (.not. all(empty > 2)) error stop 6'//new_line('a')// &
            '    print *, "ok"'//new_line('a')// &
            'end program main'

        test_runtime_rank2_comparison_mask = expect_output( &
            source, ' ok'//new_line('a'), '/tmp/ffc_mask_runtime_rank2')
    end function test_runtime_rank2_comparison_mask

    logical function test_runtime_rank3_comparison_mask()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '    integer, allocatable :: a(:,:,:), empty(:,:,:)'//new_line('a')// &
            '    allocate(a(2,2,2), empty(0,2,2))'//new_line('a')// &
            '    a(1,1,1) = 1; a(2,1,1) = 2; a(1,2,1) = 3; a(2,2,1) = 4'//new_line('a')// &
            '    a(1,1,2) = 5; a(2,1,2) = 6; a(1,2,2) = 7; a(2,2,2) = 8'//new_line('a')// &
            '    if (count(a > 4) /= 4) error stop 1'//new_line('a')// &
            '    if (.not. any(a == 7)) error stop 2'//new_line('a')// &
            '    if (.not. all(a > 0)) error stop 3'//new_line('a')// &
            '    if (count(empty > 4) /= 0) error stop 4'//new_line('a')// &
            '    if (any(empty > 4)) error stop 5'//new_line('a')// &
            '    if (.not. all(empty > 4)) error stop 6'//new_line('a')// &
            '    print *, "ok"'//new_line('a')// &
            'end program main'

        test_runtime_rank3_comparison_mask = expect_output( &
            source, ' ok'//new_line('a'), '/tmp/ffc_mask_runtime_rank3')
    end function test_runtime_rank3_comparison_mask

    logical function test_runtime_rank4_comparison_mask()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '    integer, allocatable :: a(:,:,:,:), empty(:,:,:,:)'//new_line('a')// &
            '    allocate(a(2,2,2,2), empty(0,2,2,2))'//new_line('a')// &
            '    a(1,1,1,1) = 1; a(2,1,1,1) = 2'//new_line('a')// &
            '    a(1,2,1,1) = 3; a(2,2,1,1) = 4'//new_line('a')// &
            '    a(1,1,2,1) = 5; a(2,1,2,1) = 6'//new_line('a')// &
            '    a(1,2,2,1) = 7; a(2,2,2,1) = 8'//new_line('a')// &
            '    a(1,1,1,2) = 9; a(2,1,1,2) = 10'//new_line('a')// &
            '    a(1,2,1,2) = 11; a(2,2,1,2) = 12'//new_line('a')// &
            '    a(1,1,2,2) = 13; a(2,1,2,2) = 14'//new_line('a')// &
            '    a(1,2,2,2) = 15; a(2,2,2,2) = 16'//new_line('a')// &
            '    print *, count(a > 8)'//new_line('a')// &
            '    print *, any(a == 16)'//new_line('a')// &
            '    print *, all(a > 0)'//new_line('a')// &
            '    print *, count(empty > 0)'//new_line('a')// &
            '    print *, any(empty > 0)'//new_line('a')// &
            '    print *, all(empty > 0)'//new_line('a')// &
            'end program main'

        test_runtime_rank4_comparison_mask = expect_output_matches_gfortran( &
            source, 'runtime_rank4_comparison')
    end function test_runtime_rank4_comparison_mask

end program test_session_array_mask_reduction_compiler
