program test_session_allocatable_lifecycle_compiler
    ! Lifecycle of an integer 1-D allocatable. #184 covers the descriptor
    ! declaration; allocate/deallocate and element access land in later issues.
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session allocatable lifecycle compiler test ==='

    all_passed = .true.
    if (.not. test_declare_integer_allocatable_compiles()) all_passed = .false.
    if (.not. test_declare_two_allocatables_compiles()) all_passed = .false.
    if (.not. test_real_allocatable_lifecycle()) all_passed = .false.
    if (.not. test_allocate_literal_size()) all_passed = .false.
    if (.not. test_allocate_runtime_size()) all_passed = .false.
    if (.not. test_allocate_then_deallocate()) all_passed = .false.
    if (.not. test_deallocate_unallocated_does_not_crash()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: integer allocatable descriptors lower through direct LIRIC'

contains

    logical function test_allocate_literal_size()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(5))'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_allocate_literal_size = expect_exit_status( &
            source, 0, '/tmp/ffc_alloc_lit_test')
    end function test_allocate_literal_size

    logical function test_allocate_runtime_size()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  n = 5'//new_line('a')// &
            '  allocate(a(n))'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_allocate_runtime_size = expect_exit_status( &
            source, 0, '/tmp/ffc_alloc_rt_test')
    end function test_allocate_runtime_size

    logical function test_allocate_then_deallocate()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(5))'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_allocate_then_deallocate = expect_exit_status( &
            source, 0, '/tmp/ffc_alloc_dealloc_test')
    end function test_allocate_then_deallocate

    logical function test_deallocate_unallocated_does_not_crash()
        ! MVP behaviour: deallocating an unallocated variable frees NULL (a
        ! no-op) and exits cleanly. Not standard-conforming (gfortran errors),
        ! documented in docs/RUNTIME_ABI.md.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_deallocate_unallocated_does_not_crash = expect_exit_status( &
            source, 0, '/tmp/ffc_dealloc_unalloc_test')
    end function test_deallocate_unallocated_does_not_crash

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

    logical function test_real_allocatable_lifecycle()
        ! Real 1-D allocatables compile: allocate, constructor assign, print.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real, allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  a = [1.0, 2.0, 3.0]'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_real_allocatable_lifecycle = expect_output( &
            source, &
            '   1.00000000       2.00000000       3.00000000    '// &
            new_line('a'), '/tmp/ffc_alloc_real_test')
    end function test_real_allocatable_lifecycle

end program test_session_allocatable_lifecycle_compiler
