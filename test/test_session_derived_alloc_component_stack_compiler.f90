program test_session_derived_alloc_component_stack_compiler
    ! Whole-component operations on a rank-1 allocatable array component through
    ! the direct LIRIC session: assignment from an array constructor with
    ! auto-allocation, assignment from a whole plain-array variable,
    ! sum() over the whole component, and all()/any() of an elementwise
    ! comparison against a conforming array.
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session allocatable array component stack test ==='

    all_passed = .true.
    if (.not. test_constructor_autoalloc_sum()) all_passed = .false.
    if (.not. test_plain_array_rhs()) all_passed = .false.
    if (.not. test_all_any_mask()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: allocatable array component whole operations lower'

contains

    logical function test_constructor_autoalloc_sum()
        ! x%v = [1,2,3] auto-allocates the component to the constructor extent;
        ! sum(x%v) reduces over the whole component.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer, allocatable :: v(:)'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  x%v = [1, 2, 3]'//new_line('a')// &
            '  if (.not. allocated(x%v)) error stop 1'//new_line('a')// &
            '  if (size(x%v) /= 3) error stop 2'//new_line('a')// &
            '  print *, sum(x%v)'//new_line('a')// &
            'end program main'

        test_constructor_autoalloc_sum = expect_output( &
            source, '           6'//new_line('a'), &
            '/tmp/ffc_alloc_comp_stack_ctor')
    end function test_constructor_autoalloc_sum

    logical function test_plain_array_rhs()
        ! key%value = test copies a whole plain array into the component, sizing
        ! it to the source extent; element reads confirm the copy.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: key_t'//new_line('a')// &
            '    integer, allocatable :: value(:)'//new_line('a')// &
            '  end type key_t'//new_line('a')// &
            '  type(key_t) :: key'//new_line('a')// &
            '  integer :: test(5) = [1, 2, 3, 4, 5]'//new_line('a')// &
            '  key%value = test'//new_line('a')// &
            '  if (size(key%value) /= 5) error stop 1'//new_line('a')// &
            '  if (key%value(1) /= 1) error stop 2'//new_line('a')// &
            '  if (key%value(5) /= 5) error stop 3'//new_line('a')// &
            '  print *, key%value(1) + key%value(5)'//new_line('a')// &
            'end program main'

        test_plain_array_rhs = expect_output( &
            source, '           6'//new_line('a'), &
            '/tmp/ffc_alloc_comp_stack_plain')
    end function test_plain_array_rhs

    logical function test_all_any_mask()
        ! all()/any() of an elementwise comparison between a whole allocatable
        ! array component and a conforming array.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: key_t'//new_line('a')// &
            '    integer, allocatable :: value(:)'//new_line('a')// &
            '  end type key_t'//new_line('a')// &
            '  type(key_t) :: key'//new_line('a')// &
            '  integer :: test(5) = [1, 2, 3, 4, 5]'//new_line('a')// &
            '  key%value = test'//new_line('a')// &
            '  if (.not. all(key%value == test)) error stop 1'//new_line('a')// &
            '  if (any(key%value /= test)) error stop 2'//new_line('a')// &
            'end program main'

        test_all_any_mask = expect_exit_status( &
            source, 0, '/tmp/ffc_alloc_comp_stack_mask')
    end function test_all_any_mask

end program test_session_derived_alloc_component_stack_compiler
