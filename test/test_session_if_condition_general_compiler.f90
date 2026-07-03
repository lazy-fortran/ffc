program test_session_if_condition_general_compiler
    ! Cluster W6: IF condition generality. Covers logical function results
    ! (allocated(), a contained logical function), a plain logical array
    ! element, and .not./.and. trees over those, none of which the direct
    ! LIRIC session accepted before.
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session IF condition generality compiler test ==='

    all_passed = .true.
    if (.not. test_allocated_array_condition()) all_passed = .false.
    if (.not. test_allocated_character_condition()) all_passed = .false.
    if (.not. test_logical_array_element_condition()) all_passed = .false.
    if (.not. test_contained_logical_function_condition()) all_passed = .false.
    if (.not. test_not_and_tree_over_calls()) all_passed = .false.
    if (.not. test_inline_if_without_then()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: general IF conditions lower through direct LIRIC session'

contains

    logical function test_allocated_array_condition()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  if (allocated(a)) stop 1'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  if (.not. allocated(a)) stop 2'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            '  if (allocated(a)) stop 3'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_allocated_array_condition = expect_exit_status( &
            source, 0, '/tmp/ffc_if_allocated_array_test')
    end function test_allocated_array_condition

    logical function test_allocated_character_condition()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  if (allocated(s)) stop 1'//new_line('a')// &
            '  allocate(character(len=5) :: s)'//new_line('a')// &
            '  if (.not. allocated(s)) stop 2'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_allocated_character_condition = expect_exit_status( &
            source, 0, '/tmp/ffc_if_allocated_char_test')
    end function test_allocated_character_condition

    logical function test_logical_array_element_condition()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: flags(3)'//new_line('a')// &
            '  flags = [.true., .false., .true.]'//new_line('a')// &
            '  if (.not. flags(1)) stop 1'//new_line('a')// &
            '  if (flags(2)) stop 2'//new_line('a')// &
            '  if (.not. flags(3)) stop 3'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_logical_array_element_condition = expect_exit_status( &
            source, 0, '/tmp/ffc_if_logical_array_element_test')
    end function test_logical_array_element_condition

    logical function test_contained_logical_function_condition()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  if (.not. is_even(4)) stop 1'//new_line('a')// &
            '  if (is_even(3)) stop 2'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'contains'//new_line('a')// &
            '  logical function is_even(n)'//new_line('a')// &
            '    integer, intent(in) :: n'//new_line('a')// &
            '    is_even = mod(n, 2) == 0'//new_line('a')// &
            '  end function'//new_line('a')// &
            'end program main'

        test_contained_logical_function_condition = expect_exit_status( &
            source, 0, '/tmp/ffc_if_contained_logical_fn_test')
    end function test_contained_logical_function_condition

    logical function test_not_and_tree_over_calls()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: flags(2)'//new_line('a')// &
            '  flags = [.true., .false.]'//new_line('a')// &
            '  if (.not. flags(2) .and. flags(1)) then'//new_line('a')// &
            '    print *, "combo"'//new_line('a')// &
            '  end if'//new_line('a')// &
            'end program main'

        test_not_and_tree_over_calls = expect_output( &
            source, ' combo'//new_line('a'), '/tmp/ffc_if_not_and_tree_test')
    end function test_not_and_tree_over_calls

    logical function test_inline_if_without_then()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  if (allocated(a)) deallocate(a)'//new_line('a')// &
            '  allocate(a(2))'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_inline_if_without_then = expect_exit_status( &
            source, 0, '/tmp/ffc_if_inline_no_then_test')
    end function test_inline_if_without_then

end program test_session_if_condition_general_compiler
