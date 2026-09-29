program test_session_data_implied_do_zero_compiler
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none

    print *, '=== DATA partial init zero-fills the rest (#2349) ==='

    ! Any DATA initialization of an array gives the whole object the default
    ! initial value for its type; elements the DATA list omits read zero,
    ! not stack garbage. Before this fix the uncovered elements of a fixed
    ! numeric array kept uninitialized alloca bytes, so the printed row (and
    ! the exit-deterministic stdout) diverged from gfortran.
    if (.not. expect_output_matches_gfortran( &
        'program p'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    integer :: a(4), i'//new_line('a')// &
        '    data (a(i), i = 1, 2) / 7, 8 /'//new_line('a')// &
        '    print *, a'//new_line('a')// &
        'end program p', &
        'data_implied_do_zero_i32')) stop 1

    if (.not. expect_output_matches_gfortran( &
        'program p'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    real :: arr(3, 3)'//new_line('a')// &
        '    integer :: i, j'//new_line('a')// &
        '    data ((arr(i, j), i = 1, j), j = 1, 3) /6 * 1.0/'//new_line('a')// &
        '    print *, arr'//new_line('a')// &
        'end program p', &
        'data_implied_do_zero_r32')) stop 1

    print *, 'PASS: DATA partial init zero-fills omitted elements'
end program test_session_data_implied_do_zero_compiler
