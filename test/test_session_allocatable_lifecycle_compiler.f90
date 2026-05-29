program test_session_allocatable_lifecycle_compiler
    ! Lifecycle of an integer 1-D allocatable. #184 covers the descriptor
    ! declaration; allocate/deallocate and element access land in later issues.
    use ffc_test_support, only: expect_exit_status, expect_error_contains
    implicit none

    logical :: all_passed

    print *, '=== direct session allocatable lifecycle compiler test ==='

    all_passed = .true.
    if (.not. test_declare_integer_allocatable_compiles()) all_passed = .false.
    if (.not. test_declare_two_allocatables_compiles()) all_passed = .false.
    if (.not. test_real_allocatable_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: integer allocatable descriptors lower through direct LIRIC'

contains

    logical function test_declare_integer_allocatable_compiles()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_declare_integer_allocatable_compiles = expect_exit_status( &
            source, 0, '/tmp/ffc_alloc_declare_test')
    end function test_declare_integer_allocatable_compiles

    logical function test_declare_two_allocatables_compiles()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  integer, allocatable :: b(:)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_declare_two_allocatables_compiles = expect_exit_status( &
            source, 0, '/tmp/ffc_alloc_declare2_test')
    end function test_declare_two_allocatables_compiles

    logical function test_real_allocatable_rejected()
        ! Only integer 1-D allocatables are supported; a real allocatable must
        ! still be rejected with a targeted diagnostic.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real, allocatable :: a(:)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_real_allocatable_rejected = expect_error_contains( &
            source, 'integer arrays', '/tmp/ffc_alloc_real_test')
    end function test_real_allocatable_rejected

end program test_session_allocatable_lifecycle_compiler
