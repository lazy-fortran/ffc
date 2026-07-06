program test_session_derived_nested_array_component_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    all_passed = .true.

    write(*,'(A)') ' === derived nested-array component compiler test ==='

    if (.not. test_fixed_derived_array_scalar_field()) all_passed = .false.
    if (.not. test_fixed_derived_array_default()) all_passed = .false.
    if (.not. test_nested_fixed_derived_arrays_alloc_leaf()) all_passed = .false.

    if (all_passed) then
        write(*,'(A)') ' PASS: fixed arrays of derived components lower'
    else
        write(*,'(A)') ' FAIL: fixed derived-array component lowering regressed'
        stop 1
    end if

contains

    logical function test_fixed_derived_array_scalar_field()
        ! A fixed-size array of derived components: element write, element read
        ! through the subscripted-component chain obj%arr(i)%field.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: inner_t'//new_line('a')// &
            '    integer :: id'//new_line('a')// &
            '  end type inner_t'//new_line('a')// &
            '  type :: outer_t'//new_line('a')// &
            '    type(inner_t) :: arr(3)'//new_line('a')// &
            '  end type outer_t'//new_line('a')// &
            '  type(outer_t) :: x'//new_line('a')// &
            '  x%arr(1)%id = 10'//new_line('a')// &
            '  x%arr(2)%id = 20'//new_line('a')// &
            '  x%arr(3)%id = 30'//new_line('a')// &
            '  print *, x%arr(1)%id + x%arr(2)%id + x%arr(3)%id'//new_line('a')// &
            'end program main'

        test_fixed_derived_array_scalar_field = expect_output( &
            source, '          60'//new_line('a'), &
            '/tmp/ffc_session_derived_nested_arr_field')
    end function test_fixed_derived_array_scalar_field

    logical function test_fixed_derived_array_default()
        ! Each element of a fixed derived-array component receives the inner
        ! type's component default, not just the first element.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: inner_t'//new_line('a')// &
            '    integer :: id = 7'//new_line('a')// &
            '  end type inner_t'//new_line('a')// &
            '  type :: outer_t'//new_line('a')// &
            '    type(inner_t) :: arr(3)'//new_line('a')// &
            '  end type outer_t'//new_line('a')// &
            '  type(outer_t) :: x'//new_line('a')// &
            '  print *, x%arr(1)%id + x%arr(2)%id + x%arr(3)%id'//new_line('a')// &
            'end program main'

        test_fixed_derived_array_default = expect_output( &
            source, '          21'//new_line('a'), &
            '/tmp/ffc_session_derived_nested_arr_default')
    end function test_fixed_derived_array_default

    logical function test_nested_fixed_derived_arrays_alloc_leaf()
        ! Deeply nested fixed derived-array components with an allocatable
        ! integer array leaf, as in the lfortran arrayitem_02 corpus file:
        ! allocate and size through obj%w(1)%z(2)%y(3)%x.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: a1'//new_line('a')// &
            '    integer, allocatable :: x(:)'//new_line('a')// &
            '  end type'//new_line('a')// &
            '  type :: a2'//new_line('a')// &
            '    type(a1) :: y(4)'//new_line('a')// &
            '  end type'//new_line('a')// &
            '  type :: a3'//new_line('a')// &
            '    type(a2) :: z(3)'//new_line('a')// &
            '  end type'//new_line('a')// &
            '  type :: a4'//new_line('a')// &
            '    type(a3) :: w(2)'//new_line('a')// &
            '  end type'//new_line('a')// &
            '  type(a4) :: obj'//new_line('a')// &
            '  allocate(obj%w(1)%z(2)%y(3)%x(50))'//new_line('a')// &
            '  print *, size(obj%w(1)%z(2)%y(3)%x)'//new_line('a')// &
            'end program main'

        test_nested_fixed_derived_arrays_alloc_leaf = expect_output( &
            source, '          50'//new_line('a'), &
            '/tmp/ffc_session_derived_nested_arr_leaf')
    end function test_nested_fixed_derived_arrays_alloc_leaf

end program test_session_derived_nested_array_component_compiler
