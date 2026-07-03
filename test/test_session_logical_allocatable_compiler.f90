program test_session_logical_allocatable_compiler
    ! Logical 1-D allocatable arrays through the direct LIRIC session: declare,
    ! allocate(a(n)), element write/read, constructor assign, whole-array
    ! print, and deallocate. Outputs match gfortran list-directed formatting.
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session logical allocatable compiler test ==='

    all_passed = .true.
    if (.not. test_constructor_assign_print()) all_passed = .false.
    if (.not. test_element_write_print()) all_passed = .false.
    if (.not. test_element_read_stop()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: logical allocatable arrays lower through direct LIRIC session'

contains

    logical function test_constructor_assign_print()
        ! allocate then whole-array constructor assignment and print.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical, allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  a = [.true., .false., .true.]'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_constructor_assign_print = expect_output( &
            source, ' T F T'//new_line('a'), '/tmp/ffc_logical_alloc_ctor')
    end function test_constructor_assign_print

    logical function test_element_write_print()
        ! Per-element writes, then a single-element print.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical, allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  a(1) = .true.'//new_line('a')// &
            '  a(2) = .false.'//new_line('a')// &
            '  a(3) = .true.'//new_line('a')// &
            '  print *, a(1), a(2), a(3)'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_element_write_print = expect_output( &
            source, ' T F T'//new_line('a'), '/tmp/ffc_logical_alloc_elem')
    end function test_element_write_print

    logical function test_element_read_stop()
        ! a(2) = .true.; exit with 1 when true, 0 when false, via merge.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical, allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(2))'//new_line('a')// &
            '  a(1) = .false.'//new_line('a')// &
            '  a(2) = .true.'//new_line('a')// &
            '  if (a(2)) then'//new_line('a')// &
            '    stop 1'//new_line('a')// &
            '  end if'//new_line('a')// &
            'end program main'

        test_element_read_stop = expect_exit_status( &
            source, 1, '/tmp/ffc_logical_alloc_read')
    end function test_element_read_stop

end program test_session_logical_allocatable_compiler
