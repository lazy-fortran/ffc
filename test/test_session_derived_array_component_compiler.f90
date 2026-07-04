program test_session_derived_array_component_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    all_passed = .true.

    write(*,'(A)') ' === derived array component compiler test ==='

    if (.not. test_array_component_write_read_print()) all_passed = .false.
    if (.not. test_array_component_alongside_nested()) all_passed = .false.
    if (.not. test_real_array_component()) all_passed = .false.
    if (.not. test_real8_array_component()) all_passed = .false.
    if (.not. test_logical_array_component()) all_passed = .false.
    if (.not. test_array_component_constructor()) all_passed = .false.
    if (.not. test_array_component_broadcast()) all_passed = .false.
    if (.not. test_array_component_whole_read()) all_passed = .false.
    if (.not. test_array_component_component_copy()) all_passed = .false.

    if (all_passed) then
        write(*,'(A)') ' PASS: integer/real/logical array components lower'
    else
        write(*,'(A)') ' FAIL: array component lowering regressed'
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

    logical function test_real_array_component()
        ! real(4) array component: element write, element read in arithmetic,
        ! list-directed print matching gfortran's single-precision format.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    real :: r(2)'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  x%r(1) = 1.5'//new_line('a')// &
            '  x%r(2) = 2.0'//new_line('a')// &
            '  print *, x%r(1) + x%r(2)'//new_line('a')// &
            'end program main'

        test_real_array_component = expect_output( &
            source, '   3.50000000    '//new_line('a'), &
            '/tmp/ffc_session_derived_arrcomp_real_test')
    end function test_real_array_component

    logical function test_real8_array_component()
        ! real(8) array component: each element spans two i32 slots, so element
        ! addressing must scale the index by the slot width.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    real(8) :: d(2)'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  x%d(1) = 3.25d0'//new_line('a')// &
            '  x%d(2) = 4.75d0'//new_line('a')// &
            '  print *, x%d(1) + x%d(2)'//new_line('a')// &
            'end program main'

        test_real8_array_component = expect_output( &
            source, '   8.0000000000000000     '//new_line('a'), &
            '/tmp/ffc_session_derived_arrcomp_real8_test')
    end function test_real8_array_component

    logical function test_logical_array_component()
        ! logical array component: element write, read in a condition, and
        ! list-directed print formatting as T/F.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    logical :: f(2)'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  x%f(1) = .true.'//new_line('a')// &
            '  x%f(2) = .false.'//new_line('a')// &
            '  if (x%f(1) .and. .not. x%f(2)) print *, x%f(1), x%f(2)'// &
            new_line('a')// &
            'end program main'

        test_logical_array_component = expect_output( &
            source, ' T F'//new_line('a'), &
            '/tmp/ffc_session_derived_arrcomp_logical_test')
    end function test_logical_array_component

    logical function test_array_component_constructor()
        ! Whole-component assignment from an array constructor stores each
        ! element into the component's inline slots in linear order.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer :: v(3)'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  x%v = [10, 20, 30]'//new_line('a')// &
            '  print *, x%v(1) + x%v(2) + x%v(3)'//new_line('a')// &
            'end program main'

        test_array_component_constructor = expect_output( &
            source, '          60'//new_line('a'), &
            '/tmp/ffc_session_derived_arrcomp_ctor_test')
    end function test_array_component_constructor

    logical function test_array_component_broadcast()
        ! Scalar broadcast into a whole array component stores the value into
        ! every element.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer :: v(3)'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  x%v = 7'//new_line('a')// &
            '  print *, x%v(1) + x%v(2) + x%v(3)'//new_line('a')// &
            'end program main'

        test_array_component_broadcast = expect_output( &
            source, '          21'//new_line('a'), &
            '/tmp/ffc_session_derived_arrcomp_bcast_test')
    end function test_array_component_broadcast

    logical function test_array_component_whole_read()
        ! Whole-component read: a plain array assigned the array component copies
        ! each element out through the component offset.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer :: v(3)'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  integer :: a(3)'//new_line('a')// &
            '  x%v = [10, 20, 30]'//new_line('a')// &
            '  a = x%v'//new_line('a')// &
            '  print *, a(1) + a(2) + a(3)'//new_line('a')// &
            'end program main'

        test_array_component_whole_read = expect_output( &
            source, '          60'//new_line('a'), &
            '/tmp/ffc_session_derived_arrcomp_read_test')
    end function test_array_component_whole_read

    logical function test_array_component_component_copy()
        ! Whole-component copy between two instances of the same type.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer :: v(3)'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x, y'//new_line('a')// &
            '  x%v = [10, 20, 30]'//new_line('a')// &
            '  y%v = x%v'//new_line('a')// &
            '  print *, y%v(1) + y%v(2) + y%v(3)'//new_line('a')// &
            'end program main'

        test_array_component_component_copy = expect_output( &
            source, '          60'//new_line('a'), &
            '/tmp/ffc_session_derived_arrcomp_copy_test')
    end function test_array_component_component_copy

end program test_session_derived_array_component_compiler
