program test_session_random_number
    ! RANDOM_NUMBER intrinsic subroutine (#576). Before this support existed,
    ! `call random_number(x)` lowered to a call to an undeclared external
    ! symbol: the program linked and then died at load time with
    ! "undefined symbol: random_number" (exit 127). The value is
    ! nondeterministic, so the oracles are observable invariants: the draw
    ! lies in [0,1), successive draws differ, and the argument round-trips.
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session random_number compiler test ==='

    all_passed = .true.
    if (.not. test_draw_in_unit_interval()) all_passed = .false.
    if (.not. test_successive_draws_differ()) all_passed = .false.
    if (.not. test_corpus_shape_runs()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: random_number lowers through LIRIC and runs'

contains

    logical function test_draw_in_unit_interval()
        ! 0 <= x < 1 for every draw. The guards are separate tests because
        ! .and. does not short-circuit and both operands are always valid here
        ! only by accident; keeping them nested states the intent.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: x'//new_line('a')// &
            '  call random_number(x)'//new_line('a')// &
            '  if (x < 0.0) then'//new_line('a')// &
            '    print *, "LOW"'//new_line('a')// &
            '  else if (x >= 1.0) then'//new_line('a')// &
            '    print *, "HIGH"'//new_line('a')// &
            '  else'//new_line('a')// &
            '    print *, "INRANGE"'//new_line('a')// &
            '  end if'//new_line('a')// &
            'end program main'

        test_draw_in_unit_interval = expect_output( &
            source, ' INRANGE'//new_line('a'), '/tmp/ffc_random_number_range')
    end function test_draw_in_unit_interval

    logical function test_successive_draws_differ()
        ! A constant would satisfy the range test; two draws must not be equal.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: a, b'//new_line('a')// &
            '  call random_number(a)'//new_line('a')// &
            '  call random_number(b)'//new_line('a')// &
            '  if (a == b) then'//new_line('a')// &
            '    print *, "SAME"'//new_line('a')// &
            '  else'//new_line('a')// &
            '    print *, "DIFFER"'//new_line('a')// &
            '  end if'//new_line('a')// &
            'end program main'

        test_successive_draws_differ = expect_output( &
            source, ' DIFFER'//new_line('a'), '/tmp/ffc_random_number_differ')
    end function test_successive_draws_differ

    logical function test_corpus_shape_runs()
        ! The shape of do_concurrent_3_valid.f90: the drawn value feeds a
        ! DO CONCURRENT body and is printed. It must run, not exit 127.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  real :: array(4), val'//new_line('a')// &
            '  call random_number(val)'//new_line('a')// &
            '  do concurrent(i=1:4)'//new_line('a')// &
            '    array(i) = val*real(i)'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, array(1)'//new_line('a')// &
            'end program main'

        test_corpus_shape_runs = expect_exit_status( &
            source, 0, '/tmp/ffc_random_number_corpus')
    end function test_corpus_shape_runs

end program test_session_random_number
