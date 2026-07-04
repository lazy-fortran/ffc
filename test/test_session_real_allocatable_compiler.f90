program test_session_real_allocatable_compiler
    ! Real 1-D allocatable arrays through the direct LIRIC session: declare,
    ! allocate(a(n)), element write, constructor assign, whole-array print, and
    ! deallocate. Outputs match gfortran list-directed formatting.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session real allocatable compiler test ==='

    all_passed = .true.
    if (.not. test_constructor_assign_print()) all_passed = .false.
    if (.not. test_element_write_print()) all_passed = .false.
    if (.not. test_real64_allocatable()) all_passed = .false.
    if (.not. test_runtime_size_compiles()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: real allocatable arrays lower through direct LIRIC session'

contains

    logical function test_constructor_assign_print()
        ! allocate then whole-array constructor assignment and print.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real, allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  a = [1.0, 2.0, 3.0]'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_constructor_assign_print = expect_output( &
            source, &
            '   1.00000000       2.00000000       3.00000000    '// &
            new_line('a'), '/tmp/ffc_real_alloc_ctor')
    end function test_constructor_assign_print

    logical function test_element_write_print()
        ! Per-element writes through a DO loop, then whole-array print.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real, allocatable :: a(:)'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  allocate(a(4))'//new_line('a')// &
            '  do i = 1, 4'//new_line('a')// &
            '    a(i) = real(i)'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, a(2)'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_element_write_print = expect_output( &
            source, &
            '   2.00000000    '//new_line('a')// &
            '   1.00000000       2.00000000       3.00000000    '// &
            '   4.00000000    '//new_line('a'), '/tmp/ffc_real_alloc_elem')
    end function test_element_write_print

    logical function test_real64_allocatable()
        ! real(real64) allocatable uses an 8-byte stride and f64 print.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_fortran_env, only: dp => real64'// &
            new_line('a')// &
            '  real(dp), allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(2))'//new_line('a')// &
            '  a = [1.5d0, 2.5d0]'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_real64_allocatable = expect_output( &
            source, &
            '   1.5000000000000000        2.5000000000000000     '// &
            new_line('a'), '/tmp/ffc_real_alloc_r64')
    end function test_real64_allocatable

    logical function test_runtime_size_compiles()
        ! A runtime allocate extent still allocates, writes, and reads back.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real, allocatable :: a(:)'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  n = 3'//new_line('a')// &
            '  allocate(a(n))'//new_line('a')// &
            '  a(1) = 7.0'//new_line('a')// &
            '  print *, a(1)'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_runtime_size_compiles = expect_output( &
            source, '   7.00000000    '//new_line('a'), &
            '/tmp/ffc_real_alloc_rt')
    end function test_runtime_size_compiles

end program test_session_real_allocatable_compiler
