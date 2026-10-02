! fo: dispatcher
module ffc_case_test_session_allocatable_lower_bounds_compiler
    implicit none
    private
    public :: case_test_session_allocatable_lower_bounds_compiler
    interface
        subroutine case_test_session_allocatable_lower_bounds_compiler()
        end subroutine case_test_session_allocatable_lower_bounds_compiler
    end interface
end module ffc_case_test_session_allocatable_lower_bounds_compiler

subroutine case_test_session_allocatable_lower_bounds_compiler()
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none
    save

    logical :: passed

    passed = .true.
    if (.not. test_issue762_constructor()) passed = .false.
    if (.not. test_zero_negative_integer()) passed = .false.
    if (.not. test_real_kinds()) passed = .false.
    if (.not. test_logical()) passed = .false.
    if (.not. test_dynamic_bound_variables()) passed = .false.
    if (.not. test_assumed_shape_dummy_writeback()) passed = .false.
    if (.not. test_allocatable_dummy_writeback()) passed = .false.
    if (.not. test_adjacent_empty_bounds()) passed = .false.
    if (.not. test_constructor_preserve_reset()) passed = .false.
    if (.not. test_default_bounds_control()) passed = .false.
    if (.not. passed) stop 1
    print *, 'PASS: allocated rank-one bounds and assignment semantics'

contains

    logical function test_issue762_constructor()
        character(len=*), parameter :: source = &
            'program issue762_constructor'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    integer, allocatable :: a(:)'//new_line('a')// &
            '    allocate(a(2:4))'//new_line('a')// &
            '    a = [10, 20, 30]'//new_line('a')// &
            '    print *, a(2), a(4), lbound(a, 1), ubound(a, 1), size(a)'// &
            new_line('a')// &
            'end program issue762_constructor'

        test_issue762_constructor = expect_output_matches_gfortran(source, &
            'allocated_bounds_01_issue762_constructor')
    end function test_issue762_constructor

    logical function test_zero_negative_integer()
        character(len=*), parameter :: source = &
            'program zero_negative_integer'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    integer, allocatable :: a(:), b(:)'//new_line('a')// &
            '    allocate(a(0:2))'//new_line('a')// &
            '    a(0) = 7'//new_line('a')// &
            '    a(1) = 11'//new_line('a')// &
            '    a(2) = 13'//new_line('a')// &
            '    print *, lbound(a, 1), ubound(a, 1), size(a), size(a, 1)'// &
            new_line('a')// &
            '    print *, a(0), a(1), a(2)'//new_line('a')// &
            '    allocate(b(-2:0))'//new_line('a')// &
            '    b(-2) = 17'//new_line('a')// &
            '    b(-1) = 19'//new_line('a')// &
            '    b(0) = 23'//new_line('a')// &
            '    print *, lbound(b, 1), ubound(b, 1), size(b), size(b, 1)'// &
            new_line('a')// &
            '    print *, b(-2), b(-1), b(0)'//new_line('a')// &
            'end program zero_negative_integer'

        test_zero_negative_integer = expect_output_matches_gfortran(source, &
            'allocated_bounds_02_zero_negative_integer')
    end function test_zero_negative_integer

    logical function test_real_kinds()
        character(len=*), parameter :: source = &
            'program real_kinds'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    real, allocatable :: a(:)'//new_line('a')// &
            '    real(kind=8), allocatable :: b(:)'//new_line('a')// &
            '    allocate(a(-1:1))'//new_line('a')// &
            '    a(-1) = 1.25'//new_line('a')// &
            '    a(0) = 2.50'//new_line('a')// &
            '    a(1) = -0.75'//new_line('a')// &
            '    print *, lbound(a, 1), ubound(a, 1), size(a)'//new_line('a')// &
            '    print *, int(4*a(-1)), int(4*a(0)), int(4*a(1))'//new_line('a')// &
            '    allocate(b(2:4))'//new_line('a')// &
            '    b(2) = 1.50_8'//new_line('a')// &
            '    b(3) = -2.25_8'//new_line('a')// &
            '    b(4) = 3.00_8'//new_line('a')// &
            '    print *, lbound(b, 1), ubound(b, 1), size(b)'//new_line('a')// &
            '    print *, int(4*b(2)), int(4*b(3)), int(4*b(4))'//new_line('a')// &
            'end program real_kinds'

        test_real_kinds = expect_output_matches_gfortran(source, &
            'allocated_bounds_03_real_kinds')
    end function test_real_kinds

    logical function test_logical()
        character(len=*), parameter :: source = &
            'program logical_bounds'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    logical, allocatable :: a(:)'//new_line('a')// &
            '    integer :: first, middle, last'//new_line('a')// &
            '    allocate(a(0:2))'//new_line('a')// &
            '    a(0) = .true.'//new_line('a')// &
            '    a(1) = .false.'//new_line('a')// &
            '    a(2) = .true.'//new_line('a')// &
            '    first = 0'//new_line('a')// &
            '    middle = 0'//new_line('a')// &
            '    last = 0'//new_line('a')// &
            '    if (a(0)) first = 1'//new_line('a')// &
            '    if (a(1)) middle = 1'//new_line('a')// &
            '    if (a(2)) last = 1'//new_line('a')// &
            '    print *, lbound(a, 1), ubound(a, 1), size(a)'//new_line('a')// &
            '    print *, first, middle, last'//new_line('a')// &
            'end program logical_bounds'

        test_logical = expect_output_matches_gfortran(source, &
            'allocated_bounds_04_logical')
    end function test_logical

    logical function test_dynamic_bound_variables()
        character(len=*), parameter :: source = &
            'program dynamic_bound_variables'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    integer, allocatable :: a(:)'//new_line('a')// &
            '    integer :: lower, upper, i'//new_line('a')// &
            '    lower = -3'//new_line('a')// &
            '    upper = 0'//new_line('a')// &
            '    allocate(a(lower:upper))'//new_line('a')// &
            '    do i = lower, upper'//new_line('a')// &
            '        a(i) = i + 33'//new_line('a')// &
            '    end do'//new_line('a')// &
            '    lower = 99'//new_line('a')// &
            '    upper = 100'//new_line('a')// &
            '    print *, lbound(a, 1), ubound(a, 1), size(a), size(a, 1)'// &
            new_line('a')// &
            '    print *, a(-3), a(-2), a(-1), a(0)'//new_line('a')// &
            '    a(-2) = a(-3) + a(0)'//new_line('a')// &
            '    print *, a(-2), lower, upper'//new_line('a')// &
            'end program dynamic_bound_variables'

        test_dynamic_bound_variables = expect_output_matches_gfortran(source, &
            'allocated_bounds_05_dynamic_bound_variables')
    end function test_dynamic_bound_variables

    logical function test_assumed_shape_dummy_writeback()
        character(len=*), parameter :: source = &
            'program assumed_shape_dummy_writeback'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    integer, allocatable :: a(:)'//new_line('a')// &
            '    allocate(a(2:4))'//new_line('a')// &
            '    a(2) = 10'//new_line('a')// &
            '    a(3) = 20'//new_line('a')// &
            '    a(4) = 30'//new_line('a')// &
            '    call shifted(a)'//new_line('a')// &
            '    print *, lbound(a, 1), ubound(a, 1), size(a)'//new_line('a')// &
            '    print *, a(2), a(3), a(4)'//new_line('a')// &
            '    call rebased(a)'//new_line('a')// &
            '    print *, a(2), a(3), a(4)'//new_line('a')// &
            'contains'//new_line('a')// &
            '    subroutine shifted(v)'//new_line('a')// &
            '        integer, intent(inout) :: v(-5:)'//new_line('a')// &
            '        print *, lbound(v, 1), ubound(v, 1), size(v)'//new_line('a')// &
            '        v(-5) = 101'//new_line('a')// &
            '        v(-3) = 303'//new_line('a')// &
            '    end subroutine shifted'//new_line('a')// &
            '    subroutine rebased(v)'//new_line('a')// &
            '        integer, intent(inout) :: v(:)'//new_line('a')// &
            '        print *, lbound(v, 1), ubound(v, 1), size(v)'//new_line('a')// &
            '        v(1) = v(1) + 7'//new_line('a')// &
            '        v(3) = v(3) + 9'//new_line('a')// &
            '    end subroutine rebased'//new_line('a')// &
            'end program assumed_shape_dummy_writeback'

        test_assumed_shape_dummy_writeback = expect_output_matches_gfortran(source, &
            'allocated_bounds_06_assumed_shape_dummy_writeback')
    end function test_assumed_shape_dummy_writeback

    logical function test_allocatable_dummy_writeback()
        character(len=*), parameter :: source = &
            'program allocatable_dummy_writeback'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    integer, allocatable :: a(:)'//new_line('a')// &
            '    allocate(a(2:4))'//new_line('a')// &
            '    a(2) = 10'//new_line('a')// &
            '    a(3) = 20'//new_line('a')// &
            '    a(4) = 30'//new_line('a')// &
            '    call mutate(a)'//new_line('a')// &
            '    print *, lbound(a, 1), ubound(a, 1), size(a)'//new_line('a')// &
            '    print *, a(2), a(3), a(4)'//new_line('a')// &
            'contains'//new_line('a')// &
            '    subroutine mutate(v)'//new_line('a')// &
            '        integer, allocatable, intent(inout) :: v(:)'//new_line('a')// &
            '        print *, lbound(v, 1), ubound(v, 1), size(v)'//new_line('a')// &
            '        v(2) = 42'//new_line('a')// &
            '        v(4) = 84'//new_line('a')// &
            '    end subroutine mutate'//new_line('a')// &
            'end program allocatable_dummy_writeback'

        test_allocatable_dummy_writeback = expect_output_matches_gfortran(source, &
            'allocated_bounds_07_allocatable_dummy_writeback')
    end function test_allocatable_dummy_writeback

    logical function test_adjacent_empty_bounds()
        character(len=*), parameter :: source = &
            'program empty_bounds'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    integer, allocatable :: a(:)'//new_line('a')// &
            '    allocate(a(2:1))'//new_line('a')// &
            '    print *, lbound(a, 1), ubound(a, 1), size(a), size(a, 1)'// &
            new_line('a')// &
            'end program empty_bounds'

        test_adjacent_empty_bounds = expect_output_matches_gfortran(source, &
            'allocated_bounds_09_adjacent_empty_bounds')
    end function test_adjacent_empty_bounds

    logical function test_constructor_preserve_reset()
        character(len=*), parameter :: source = &
            'program constructor_preserve_reset'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    integer, allocatable :: a(:)'//new_line('a')// &
            '    allocate(a(2:4))'//new_line('a')// &
            '    a(2) = 10'//new_line('a')// &
            '    a(3) = 20'//new_line('a')// &
            '    a(4) = 30'//new_line('a')// &
            '    a = [a(4), a(2), a(3)]'//new_line('a')// &
            '    print *, lbound(a, 1), ubound(a, 1), size(a)'//new_line('a')// &
            '    print *, a(2), a(3), a(4)'//new_line('a')// &
            '    a = [1, 2]'//new_line('a')// &
            '    print *, lbound(a, 1), ubound(a, 1), size(a)'//new_line('a')// &
            '    print *, a(1), a(2)'//new_line('a')// &
            'end program constructor_preserve_reset'

        test_constructor_preserve_reset = expect_output_matches_gfortran(source, &
            'allocated_bounds_10_constructor_preserve_reset')
    end function test_constructor_preserve_reset

    logical function test_default_bounds_control()
        character(len=*), parameter :: source = &
            'program default_bounds_control'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    integer, allocatable :: a(:)'//new_line('a')// &
            '    allocate(a(3))'//new_line('a')// &
            '    a = [10, 20, 30]'//new_line('a')// &
            '    print *, lbound(a, 1), ubound(a, 1), size(a)'//new_line('a')// &
            '    print *, a(1), a(2), a(3)'//new_line('a')// &
            '    deallocate(a)'//new_line('a')// &
            '    allocate(a(1:3))'//new_line('a')// &
            '    a = [40, 50, 60]'//new_line('a')// &
            '    print *, lbound(a, 1), ubound(a, 1), size(a)'//new_line('a')// &
            '    print *, a(1), a(2), a(3)'//new_line('a')// &
            'end program default_bounds_control'

        test_default_bounds_control = expect_output_matches_gfortran(source, &
            'allocated_bounds_default_control')
    end function test_default_bounds_control

end subroutine case_test_session_allocatable_lower_bounds_compiler
