program test_session_random_seed
    ! RANDOM_SEED intrinsic subroutine (#588). Before this support existed,
    ! `call random_seed(...)` fell through to the generic external-call path
    ! and emitted a call to an undeclared symbol: the link failed with
    ! "undefined reference to `random_seed'", so no executable was produced.
    ! The oracles are observable invariants of the generated program: the
    ! reported seed size is positive, PUT/GET round-trips, and re-seeding
    ! with the same value replays the same draw.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session random_seed compiler test ==='

    all_passed = .true.
    if (.not. test_default_seed_runs()) all_passed = .false.
    if (.not. test_size_is_positive()) all_passed = .false.
    if (.not. test_put_get_round_trip()) all_passed = .false.
    if (.not. test_put_replays_sequence()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: random_seed lowers through LIRIC and runs'

contains

    logical function test_default_seed_runs()
        ! `call random_seed()` with no arguments must run and leave the
        ! generator usable: the next draw still lies in [0,1).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x'//new_line('a')// &
            '  call random_seed()'//new_line('a')// &
            '  call random_number(x)'//new_line('a')// &
            '  if (x < 0.0) then'//new_line('a')// &
            '    print *, "LOW"'//new_line('a')// &
            '  else if (x >= 1.0) then'//new_line('a')// &
            '    print *, "HIGH"'//new_line('a')// &
            '  else'//new_line('a')// &
            '    print *, "INRANGE"'//new_line('a')// &
            '  end if'//new_line('a')// &
            'end program main'

        test_default_seed_runs = expect_output( &
            source, ' INRANGE'//new_line('a'), '/tmp/ffc_random_seed_default')
    end function test_default_seed_runs

    logical function test_size_is_positive()
        ! random_seed_01.f90's oracle: the size the processor reports must be
        ! positive, so a caller can allocate a seed array of that size.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  n = -1'//new_line('a')// &
            '  call random_seed(size=n)'//new_line('a')// &
            '  if (n > 0) then'//new_line('a')// &
            '    print *, "POSITIVE"'//new_line('a')// &
            '  else'//new_line('a')// &
            '    print *, "NONPOSITIVE"'//new_line('a')// &
            '  end if'//new_line('a')// &
            'end program main'

        test_size_is_positive = expect_output( &
            source, ' POSITIVE'//new_line('a'), '/tmp/ffc_random_seed_size')
    end function test_size_is_positive

    logical function test_put_get_round_trip()
        ! What was PUT comes back out of GET.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: seed(1), got(1)'//new_line('a')// &
            '  seed(1) = 4711'//new_line('a')// &
            '  got(1) = 0'//new_line('a')// &
            '  call random_seed(put=seed)'//new_line('a')// &
            '  call random_seed(get=got)'//new_line('a')// &
            '  if (got(1) == 4711) then'//new_line('a')// &
            '    print *, "ROUNDTRIP"'//new_line('a')// &
            '  else'//new_line('a')// &
            '    print *, "LOST"'//new_line('a')// &
            '  end if'//new_line('a')// &
            'end program main'

        test_put_get_round_trip = expect_output( &
            source, ' ROUNDTRIP'//new_line('a'), '/tmp/ffc_random_seed_roundtrip')
    end function test_put_get_round_trip

    logical function test_put_replays_sequence()
        ! intrinsics_429.f90's oracle: PUT of the same seed restarts the same
        ! sequence, so two draws taken after identical PUTs are equal, while a
        ! draw taken without re-seeding differs from them.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: seed(1)'//new_line('a')// &
            '  real :: a, b, c'//new_line('a')// &
            '  seed(1) = 12345'//new_line('a')// &
            '  call random_seed(put=seed)'//new_line('a')// &
            '  call random_number(a)'//new_line('a')// &
            '  call random_seed(put=seed)'//new_line('a')// &
            '  call random_number(b)'//new_line('a')// &
            '  call random_number(c)'//new_line('a')// &
            '  if (a /= b) then'//new_line('a')// &
            '    print *, "NOTREPLAYED"'//new_line('a')// &
            '  else if (b == c) then'//new_line('a')// &
            '    print *, "STUCK"'//new_line('a')// &
            '  else'//new_line('a')// &
            '    print *, "REPLAYED"'//new_line('a')// &
            '  end if'//new_line('a')// &
            'end program main'

        test_put_replays_sequence = expect_output( &
            source, ' REPLAYED'//new_line('a'), '/tmp/ffc_random_seed_replay')
    end function test_put_replays_sequence

end program test_session_random_seed
