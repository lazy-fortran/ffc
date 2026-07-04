program test_session_runtime_bound_array_compiler
    ! Rank-1 allocatable arrays sized by a runtime value: allocate(a(n)) with n
    ! only known at runtime, then element fill, whole-array print, size, sum,
    ! and deallocate. The extent comes from the descriptor rather than a
    ! compile-time constant. Also covers an automatic local array a(n) in a
    ! procedure sized by a dummy.
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session runtime-bound array compiler test ==='

    all_passed = .true.
    if (.not. test_runtime_alloc_fill_print()) all_passed = .false.
    if (.not. test_runtime_alloc_sum_and_size()) all_passed = .false.
    if (.not. test_runtime_alloc_real_print()) all_passed = .false.
    if (.not. test_automatic_array_in_procedure()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: runtime-bound arrays lower through direct LIRIC'

contains

    logical function test_runtime_alloc_fill_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  integer :: n, i'//new_line('a')// &
            '  n = 4'//new_line('a')// &
            '  allocate(a(n))'//new_line('a')// &
            '  do i = 1, n'//new_line('a')// &
            '    a(i) = i * 2'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_runtime_alloc_fill_print = expect_output( &
            source, &
            '           2           4           6           8'//new_line('a'), &
            '/tmp/ffc_runtime_alloc_fill_print_test')
    end function test_runtime_alloc_fill_print

    logical function test_runtime_alloc_sum_and_size()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  integer :: n, i'//new_line('a')// &
            '  n = 5'//new_line('a')// &
            '  allocate(a(n))'//new_line('a')// &
            '  do i = 1, n'//new_line('a')// &
            '    a(i) = i'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, size(a)'//new_line('a')// &
            '  print *, sum(a)'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'

        test_runtime_alloc_sum_and_size = expect_output( &
            source, &
            '           5'//new_line('a')//'          15'//new_line('a'), &
            '/tmp/ffc_runtime_alloc_sum_size_test')
    end function test_runtime_alloc_sum_and_size

    logical function test_runtime_alloc_real_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real, allocatable :: r(:)'//new_line('a')// &
            '  integer :: n, i'//new_line('a')// &
            '  n = 3'//new_line('a')// &
            '  allocate(r(n))'//new_line('a')// &
            '  do i = 1, n'//new_line('a')// &
            '    r(i) = real(i) + 0.5'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            '  deallocate(r)'//new_line('a')// &
            'end program main'

        test_runtime_alloc_real_print = expect_output( &
            source, &
            '   1.50000000       2.50000000       3.50000000    '// &
            new_line('a'), &
            '/tmp/ffc_runtime_alloc_real_print_test')
    end function test_runtime_alloc_real_print

    logical function test_automatic_array_in_procedure()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: r'//new_line('a')// &
            '  call work(5, r)'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine work(n, out)'//new_line('a')// &
            '    integer, intent(in) :: n'//new_line('a')// &
            '    integer, intent(out) :: out'//new_line('a')// &
            '    integer :: tmp(n), i'//new_line('a')// &
            '    do i = 1, n'//new_line('a')// &
            '      tmp(i) = i'//new_line('a')// &
            '    end do'//new_line('a')// &
            '    out = sum(tmp)'//new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'

        test_automatic_array_in_procedure = expect_output( &
            source, '          15'//new_line('a'), &
            '/tmp/ffc_runtime_automatic_array_test')
    end function test_automatic_array_in_procedure

end program test_session_runtime_bound_array_compiler
