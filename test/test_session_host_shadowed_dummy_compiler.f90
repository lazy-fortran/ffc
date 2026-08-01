program test_session_host_shadowed_dummy
    !! A contained procedure whose scalar dummy shadows a host array must keep
    !! the host array's declaration intact, however deeply the dummy is used
    !! inside nested ASSOCIATE constructs (#584).
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session host-shadowed dummy compiler test ==='

    all_passed = .true.
    if (.not. test_nested_associate_shadowing_dummy()) all_passed = .false.
    if (.not. test_single_associate_shadowing_dummy()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: shadowed dummies stay local to the contained procedure'

contains

    logical function test_nested_associate_shadowing_dummy()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'implicit none'//new_line('a')// &
            'real :: x(4), y(4)'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'x = [1.0, 2.0, 3.0, 4.0]'//new_line('a')// &
            'do i = 1, 4'//new_line('a')// &
            '    y(i) = compute(x(i))'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, y(4)'//new_line('a')// &
            'contains'//new_line('a')// &
            '    pure function compute(x) result(r)'//new_line('a')// &
            '        real, intent(in) :: x'//new_line('a')// &
            '        real :: r'//new_line('a')// &
            '        associate (n => 1)'//new_line('a')// &
            '            associate (z => x + real(n))'//new_line('a')// &
            '                r = z'//new_line('a')// &
            '            end associate'//new_line('a')// &
            '        end associate'//new_line('a')// &
            '    end function compute'//new_line('a')// &
            'end program main'

        test_nested_associate_shadowing_dummy = expect_output( &
            source, '   5.00000000    '//new_line('a'), &
            '/tmp/ffc_session_host_shadow_nested')
    end function test_nested_associate_shadowing_dummy

    logical function test_single_associate_shadowing_dummy()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'implicit none'//new_line('a')// &
            'real :: x(4)'//new_line('a')// &
            'x = [1.0, 2.0, 3.0, 4.0]'//new_line('a')// &
            'print *, compute(x(2))'//new_line('a')// &
            'contains'//new_line('a')// &
            '    pure function compute(x) result(r)'//new_line('a')// &
            '        real, intent(in) :: x'//new_line('a')// &
            '        real :: r'//new_line('a')// &
            '        associate (z => x + 1.0)'//new_line('a')// &
            '            r = z'//new_line('a')// &
            '        end associate'//new_line('a')// &
            '    end function compute'//new_line('a')// &
            'end program main'

        test_single_associate_shadowing_dummy = expect_output( &
            source, '   3.00000000    '//new_line('a'), &
            '/tmp/ffc_session_host_shadow_single')
    end function test_single_associate_shadowing_dummy

end program test_session_host_shadowed_dummy
