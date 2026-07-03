program test_session_whole_array_minmax_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session whole-array max/min compiler test ==='

    all_passed = .true.
    if (.not. test_integer_max_min()) all_passed = .false.
    if (.not. test_real_max_min()) all_passed = .false.
    if (.not. test_allocatable_max()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: whole-array max/min lower elementwise through direct LIRIC'

contains

    logical function test_integer_max_min()
        ! Index 3 is an exact tie (both 3): max/min pick either operand
        ! there, so a select_value branch-and-phi (as WHERE already uses)
        ! is correct regardless of which side the tie resolves toward.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4), b(4), hi(4), lo(4)'//new_line('a')// &
            '  a = [1, 5, 3, -7]'//new_line('a')// &
            '  b = [2, 5, 3, 9]'//new_line('a')// &
            '  hi = max(a, b)'//new_line('a')// &
            '  lo = min(a, b)'//new_line('a')// &
            '  if (hi(1) /= 2 .or. hi(2) /= 5 .or. hi(3) /= 3 .or. &'// &
            new_line('a')// &
            '      hi(4) /= 9) error stop 1'//new_line('a')// &
            '  if (lo(1) /= 1 .or. lo(2) /= 5 .or. lo(3) /= 3 .or. &'// &
            new_line('a')// &
            '      lo(4) /= -7) error stop 2'//new_line('a')// &
            'end program main'

        test_integer_max_min = expect_exit_status( &
            source, 0, '/tmp/ffc_whole_array_minmax_int')
    end function test_integer_max_min

    logical function test_real_max_min()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real(8) :: b(3), c(3), hi(3), lo(3)'//new_line('a')// &
            '  b = [1.0d0, -5.0d0, 3.0d0]'//new_line('a')// &
            '  c = [4.0d0, 5.0d0, -1.0d0]'//new_line('a')// &
            '  hi = max(b, c)'//new_line('a')// &
            '  lo = min(b, c)'//new_line('a')// &
            '  if (hi(1) /= 4.0d0 .or. hi(2) /= 5.0d0 .or. &'//new_line('a')// &
            '      hi(3) /= 3.0d0) error stop 1'//new_line('a')// &
            '  if (lo(1) /= 1.0d0 .or. lo(2) /= -5.0d0 .or. &'//new_line('a')// &
            '      lo(3) /= -1.0d0) error stop 2'//new_line('a')// &
            'end program main'

        test_real_max_min = expect_exit_status( &
            source, 0, '/tmp/ffc_whole_array_minmax_real')
    end function test_real_max_min

    logical function test_allocatable_max()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  integer :: b(2) = [2, 2]'//new_line('a')// &
            '  allocate (a(2))'//new_line('a')// &
            '  a = [1, 2]'//new_line('a')// &
            '  a = max(a, b)'//new_line('a')// &
            '  if (a(1) /= 2 .or. a(2) /= 2) error stop 1'//new_line('a')// &
            'end program main'

        test_allocatable_max = expect_exit_status( &
            source, 0, '/tmp/ffc_whole_array_minmax_alloc')
    end function test_allocatable_max

end program test_session_whole_array_minmax_compiler
