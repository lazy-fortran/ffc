program test_session_container_contained_fn
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== program-with-module contained function compiler test ==='

    all_passed = .true.
    if (.not. test_real_contained_function()) all_passed = .false.
    if (.not. test_logical_contained_function()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: contained functions resolve in a multi-unit container'

contains

    logical function test_real_contained_function()
        ! A module preceding the program makes FortFront wrap both units in a
        ! multi_unit_container; the program's own CONTAINS function must still
        ! resolve as a contained call rather than an unsupported one.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: y'//new_line('a')// &
            '  y = bump(6.0)'//new_line('a')// &
            '  stop int(y)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  real function bump(a) result(b)'//new_line('a')// &
            '    real, intent(in) :: a'//new_line('a')// &
            '    b = a + 1.0'//new_line('a')// &
            '  end function bump'//new_line('a')// &
            'end program main'

        test_real_contained_function = expect_exit_status( &
                source, 7, '/tmp/ffc_session_container_real_fn_test')
        end function test_real_contained_function

        logical function test_logical_contained_function()
            character(len=*), parameter :: source = &
                'module m'//new_line('a')// &
                '  implicit none'//new_line('a')// &
                'end module m'//new_line('a')// &
                'program main'//new_line('a')// &
                '  implicit none'//new_line('a')// &
                '  if (is_pos(3.0)) stop 5'//new_line('a')// &
                '  stop 0'//new_line('a')// &
                'contains'//new_line('a')// &
                '  logical function is_pos(a) result(r)'//new_line('a')// &
                '    real, intent(in) :: a'//new_line('a')// &
                '    r = a > 0.0'//new_line('a')// &
                '  end function is_pos'//new_line('a')// &
                'end program main'

            test_logical_contained_function = expect_exit_status( &
                    source, 5, '/tmp/ffc_session_container_logical_fn_test')
            end function test_logical_contained_function

        end program test_session_container_contained_fn
