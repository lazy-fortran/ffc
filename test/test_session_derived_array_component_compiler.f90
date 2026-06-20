program test_session_derived_array_component_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    all_passed = .true.

    write(*,'(A)') ' === derived integer array component compiler test ==='

    if (.not. test_array_component_write_read_print()) all_passed = .false.
    if (.not. test_array_component_alongside_nested()) all_passed = .false.

    if (all_passed) then
        write(*,'(A)') ' PASS: integer array components lower through direct LIRIC'
    else
        write(*,'(A)') ' FAIL: integer array component lowering regressed'
        stop 1
    end if

contains

    logical function test_array_component_write_read_print()
        ! Element write, element read, and printing a sum of elements. The
        ! expected text matches gfortran's list-directed integer format.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer :: v(3)'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  x%v(1) = 10'//new_line('a')// &
            '  x%v(2) = 20'//new_line('a')// &
            '  x%v(3) = 30'//new_line('a')// &
            '  print *, x%v(1) + x%v(2) + x%v(3)'//new_line('a')// &
            'end program main'

        test_array_component_write_read_print = expect_output( &
            source, '          60'//new_line('a'), &
            '/tmp/ffc_session_derived_arrcomp_test')
    end function test_array_component_write_read_print

    logical function test_array_component_alongside_nested()
        ! An array component sharing a type with a scalar nested derived
        ! component: both occupy distinct, correctly sized slot spans.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: inner_t'//new_line('a')// &
            '    integer :: value'//new_line('a')// &
            '  end type inner_t'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer :: v(2)'//new_line('a')// &
            '    type(inner_t) :: c'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  x%v(1) = 3'//new_line('a')// &
            '  x%v(2) = 4'//new_line('a')// &
            '  x%c%value = 5'//new_line('a')// &
            '  print *, x%v(1) + x%v(2) + x%c%value'//new_line('a')// &
            'end program main'

        test_array_component_alongside_nested = expect_output( &
            source, '          12'//new_line('a'), &
            '/tmp/ffc_session_derived_arrcomp_nested_test')
    end function test_array_component_alongside_nested

end program test_session_derived_array_component_compiler
