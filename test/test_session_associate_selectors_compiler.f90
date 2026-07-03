program test_session_associate_selectors
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session associate array/component selector compiler test ==='

    all_passed = .true.
    if (.not. test_associate_array_section_read()) all_passed = .false.
    if (.not. test_associate_array_section_write()) all_passed = .false.
    if (.not. test_associate_component_read()) all_passed = .false.
    if (.not. test_associate_component_write()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: associate array-section and component selectors lower ' // &
        'through direct LIRIC'

contains

    logical function test_associate_array_section_read()
        ! associate (x => a(2:4)): x is a rank-1 view of a's own storage,
        ! reindexed to lower bound 1.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: a(5)'//new_line('a')// &
            'a = [1, 2, 3, 4, 5]'//new_line('a')// &
            'associate (x => a(2:4))'//new_line('a')// &
            '    if (size(x) /= 3) stop 1'//new_line('a')// &
            '    stop x(1) + x(2) + x(3)'//new_line('a')// &
            'end associate'//new_line('a')// &
            'end program main'

        test_associate_array_section_read = expect_exit_status( &
            source, 9, &
            '/tmp/ffc_session_associate_section_read')
    end function test_associate_array_section_read

    logical function test_associate_array_section_write()
        ! A write through the associate name flows back to the source array.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: a(5)'//new_line('a')// &
            'a = [1, 2, 3, 4, 5]'//new_line('a')// &
            'associate (x => a(2:4))'//new_line('a')// &
            '    x(2) = 99'//new_line('a')// &
            'end associate'//new_line('a')// &
            'stop a(3)'//new_line('a')// &
            'end program main'

        test_associate_array_section_write = expect_exit_status( &
            source, 99, &
            '/tmp/ffc_session_associate_section_write')
    end function test_associate_array_section_write

    logical function test_associate_component_read()
        ! associate (s => a%comp) aliases the component's own storage.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'type :: point_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            'end type point_t'//new_line('a')// &
            'type(point_t) :: a'//new_line('a')// &
            'a%x = 5'//new_line('a')// &
            'associate (s => a%x)'//new_line('a')// &
            '    stop s'//new_line('a')// &
            'end associate'//new_line('a')// &
            'end program main'

        test_associate_component_read = expect_exit_status( &
            source, 5, &
            '/tmp/ffc_session_associate_component_read')
    end function test_associate_component_read

    logical function test_associate_component_write()
        ! A write through the associate name flows back to the component.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'type :: point_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            'end type point_t'//new_line('a')// &
            'type(point_t) :: a'//new_line('a')// &
            'a%x = 5'//new_line('a')// &
            'associate (s => a%x)'//new_line('a')// &
            '    s = 42'//new_line('a')// &
            'end associate'//new_line('a')// &
            'stop a%x'//new_line('a')// &
            'end program main'

        test_associate_component_write = expect_exit_status( &
            source, 42, &
            '/tmp/ffc_session_associate_component_write')
    end function test_associate_component_write

end program test_session_associate_selectors
