program test_session_derived_real_component_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session derived real/logical component test ==='

    all_passed = .true.
    if (.not. test_real32_components_print()) all_passed = .false.
    if (.not. test_real64_components_print()) all_passed = .false.
    if (.not. test_logical_component_print()) all_passed = .false.
    if (.not. test_mixed_components_print()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: real and logical derived components lower through direct LIRIC'

contains

    logical function test_real32_components_print()
        ! Two default-real (f32) components, each one i32 slot: store and print.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: point_t'//new_line('a')// &
            '    real :: x'//new_line('a')// &
            '    real :: y'//new_line('a')// &
            '  end type point_t'//new_line('a')// &
            '  type(point_t) :: p'//new_line('a')// &
            '  p%x = 3.0'//new_line('a')// &
            '  p%y = 5.0'//new_line('a')// &
            '  print *, p%x'//new_line('a')// &
            '  print *, p%y'//new_line('a')// &
            'end program main'

        test_real32_components_print = expect_output( &
            source, '   3.00000000    '//new_line('a')// &
            '   5.00000000    '//new_line('a'), &
            '/tmp/ffc_derived_real32_component_test')
    end function test_real32_components_print

    logical function test_real64_components_print()
        ! A real(real64) component spans two i32 slots; the second component must
        ! still land at the right offset. Both round-trip through f64 store/load.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_fortran_env, only: real64'//new_line('a')// &
            '  type :: vec_t'//new_line('a')// &
            '    real(real64) :: a'//new_line('a')// &
            '    real(real64) :: b'//new_line('a')// &
            '  end type vec_t'//new_line('a')// &
            '  type(vec_t) :: v'//new_line('a')// &
            '  v%a = 10.0_real64'//new_line('a')// &
            '  v%b = 4.0_real64'//new_line('a')// &
            '  print *, v%a'//new_line('a')// &
            '  print *, v%b'//new_line('a')// &
            'end program main'

        test_real64_components_print = expect_output( &
            source, '   10.000000000000000     '//new_line('a')// &
            '   4.0000000000000000     '//new_line('a'), &
            '/tmp/ffc_derived_real64_component_test')
    end function test_real64_components_print

    logical function test_logical_component_print()
        ! A logical component stores in one i32 slot and prints as T/F.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: flag_t'//new_line('a')// &
            '    logical :: on'//new_line('a')// &
            '  end type flag_t'//new_line('a')// &
            '  type(flag_t) :: f'//new_line('a')// &
            '  f%on = .true.'//new_line('a')// &
            '  print *, f%on'//new_line('a')// &
            'end program main'

        test_logical_component_print = expect_output( &
            source, ' T'//new_line('a'), &
            '/tmp/ffc_derived_logical_component_test')
    end function test_logical_component_print

    logical function test_mixed_components_print()
        ! Integer and real components in one type print with their own formats,
        ! proving the per-component kind drives the print format.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: rec_t'//new_line('a')// &
            '    integer :: n'//new_line('a')// &
            '    real :: x'//new_line('a')// &
            '  end type rec_t'//new_line('a')// &
            '  type(rec_t) :: r'//new_line('a')// &
            '  r%n = 42'//new_line('a')// &
            '  r%x = 1.5'//new_line('a')// &
            '  print *, r%n'//new_line('a')// &
            '  print *, r%x'//new_line('a')// &
            'end program main'

        test_mixed_components_print = expect_output( &
            source, '          42'//new_line('a')// &
            '   1.50000000    '//new_line('a'), &
            '/tmp/ffc_derived_mixed_component_test')
    end function test_mixed_components_print

end program test_session_derived_real_component_compiler
