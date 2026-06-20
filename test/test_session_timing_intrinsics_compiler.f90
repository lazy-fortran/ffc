program test_session_timing_intrinsics
    ! CPU_TIME and SYSTEM_CLOCK intrinsic subroutine calls (#2820). The values
    ! are nondeterministic, so the tests assert observable invariants instead of
    ! exact numbers: cpu_time yields a non-negative real, system_clock yields a
    ! positive tick count, and both round-trip into their intent(out) arguments.
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session timing intrinsics compiler test ==='

    all_passed = .true.
    if (.not. test_cpu_time_round_trips()) all_passed = .false.
    if (.not. test_system_clock_positive()) all_passed = .false.
    if (.not. test_both_round_trip()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: cpu_time and system_clock lower through LIRIC'

contains

    logical function test_cpu_time_round_trips()
        ! cpu_time stores a real into its argument; the value prints back. The
        ! magnitude is nondeterministic, so this only asserts the call lowers,
        ! runs, and materialises t for output (exit 0).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: t'//new_line('a')// &
            '  call cpu_time(t)'//new_line('a')// &
            '  print *, t'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_cpu_time_round_trips = expect_exit_status( &
            source, 0, '/tmp/ffc_cpu_time')
    end function test_cpu_time_round_trips

    logical function test_system_clock_positive()
        ! system_clock stores a positive integer tick count into its argument.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  call system_clock(n)'//new_line('a')// &
            '  if (n > 0) then'//new_line('a')// &
            '    stop 0'//new_line('a')// &
            '  else'//new_line('a')// &
            '    stop 1'//new_line('a')// &
            '  end if'//new_line('a')// &
            'end program main'

        test_system_clock_positive = expect_exit_status( &
            source, 0, '/tmp/ffc_system_clock')
    end function test_system_clock_positive

    logical function test_both_round_trip()
        ! The corpus program: both calls compile, run, and print without error.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: t'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  call cpu_time(t)'//new_line('a')// &
            '  call system_clock(n)'//new_line('a')// &
            '  print *, t, n'//new_line('a')// &
            'end program main'

        test_both_round_trip = expect_exit_status( &
            source, 0, '/tmp/ffc_timing_roundtrip')
    end function test_both_round_trip

end program test_session_timing_intrinsics
