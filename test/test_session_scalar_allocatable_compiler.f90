program test_session_scalar_allocatable_compiler
    ! Scalar integer/real/logical allocatables (W11): declare, allocate,
    ! assign, deallocate, and allocate(x, source=<expr>)/mold=<expr>.
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session scalar allocatable compiler test ==='

    all_passed = .true.
    if (.not. test_declare_scalar_allocatable_compiles()) all_passed = .false.
    if (.not. test_allocate_assign_read()) all_passed = .false.
    if (.not. test_allocate_deallocate()) all_passed = .false.
    if (.not. test_real_scalar_allocatable()) all_passed = .false.
    if (.not. test_logical_scalar_allocatable()) all_passed = .false.
    if (.not. test_scalar_allocate_source_literal()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: scalar allocatables lower through direct LIRIC'

contains

    logical function test_declare_scalar_allocatable_compiles()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: x'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_declare_scalar_allocatable_compiles = expect_exit_status( &
            source, 0, '/tmp/ffc_scalar_alloc_declare_test')
    end function test_declare_scalar_allocatable_compiles

    logical function test_allocate_assign_read()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: x'//new_line('a')// &
            '  allocate(x)'//new_line('a')// &
            '  x = 10'//new_line('a')// &
            '  print *, x'//new_line('a')// &
            'end program main'

        test_allocate_assign_read = expect_output( &
            source, '          10'//new_line('a'), &
            '/tmp/ffc_scalar_alloc_rw_test')
    end function test_allocate_assign_read

    logical function test_allocate_deallocate()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: x'//new_line('a')// &
            '  allocate(x)'//new_line('a')// &
            '  x = 3'//new_line('a')// &
            '  deallocate(x)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_allocate_deallocate = expect_exit_status( &
            source, 0, '/tmp/ffc_scalar_alloc_dealloc_test')
    end function test_allocate_deallocate

    logical function test_real_scalar_allocatable()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real, allocatable :: r'//new_line('a')// &
            '  allocate(r)'//new_line('a')// &
            '  r = 4.4'//new_line('a')// &
            '  if (r /= 4.4) error stop'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_real_scalar_allocatable = expect_exit_status( &
            source, 0, '/tmp/ffc_scalar_alloc_real_test')
    end function test_real_scalar_allocatable

    logical function test_logical_scalar_allocatable()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical, allocatable :: l'//new_line('a')// &
            '  allocate(l)'//new_line('a')// &
            '  l = .true.'//new_line('a')// &
            '  if (l .neqv. .true.) error stop'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_logical_scalar_allocatable = expect_exit_status( &
            source, 0, '/tmp/ffc_scalar_alloc_logical_test')
    end function test_logical_scalar_allocatable

    logical function test_scalar_allocate_source_literal()
        ! allocate(x, source=5): the scalar mold/source form with a non-
        ! identifier source expression (#265 W11).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: x'//new_line('a')// &
            '  allocate(x, source = 5)'//new_line('a')// &
            '  if (x /= 5) error stop'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_scalar_allocate_source_literal = expect_exit_status( &
            source, 0, '/tmp/ffc_scalar_alloc_source_test')
    end function test_scalar_allocate_source_literal

end program test_session_scalar_allocatable_compiler
