program test_session_allocatable_constructor
    ! Auto-reallocation on array constructor assignment (#244 slice B2c).
    ! Assigns [e1, e2, ...] to an integer 1-D allocatable: frees old data,
    ! allocates fresh storage, fills in order, then reads back via element access.
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session allocatable constructor compiler test ==='

    all_passed = .true.
    if (.not. test_constructor_assign_and_read()) all_passed = .false.
    if (.not. test_constructor_reassign()) all_passed = .false.
    if (.not. test_identifier_copy()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: allocatable constructor assignment lowers through LIRIC'

contains

    logical function test_constructor_assign_and_read()
        ! a = [10, 20, 30]; stop a(2) -> exit status 20.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  a = [10, 20, 30]'//new_line('a')// &
            '  stop a(2)'//new_line('a')// &
            'end program main'

        test_constructor_assign_and_read = expect_exit_status( &
            source, 20, '/tmp/ffc_alloc_ctor_read')
    end function test_constructor_assign_and_read

    logical function test_constructor_reassign()
        ! Reassign to a different size; a(1) of the new [5, 6] should be 5.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  a = [10, 20, 30]'//new_line('a')// &
            '  a = [5, 6]'//new_line('a')// &
            '  stop a(1)'//new_line('a')// &
            'end program main'

        test_constructor_reassign = expect_exit_status( &
            source, 5, '/tmp/ffc_alloc_ctor_reassign')
    end function test_constructor_reassign

    logical function test_identifier_copy()
        ! a = b, both allocated rank-1 allocatables: elementwise copy.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  integer, allocatable :: b(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  allocate(b(3))'//new_line('a')// &
            '  b = [10, 20, 30]'//new_line('a')// &
            '  a = b'//new_line('a')// &
            '  stop a(2)'//new_line('a')// &
            'end program main'

        test_identifier_copy = expect_exit_status( &
            source, 20, '/tmp/ffc_alloc_ctor_copy')
    end function test_identifier_copy

end program test_session_allocatable_constructor
