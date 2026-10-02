! fo: dispatcher
module ffc_case_test_session_data_implied_do_zero_compiler
    implicit none
    private
    public :: case_test_session_data_implied_do_zero_compiler
    interface
        subroutine case_test_session_data_implied_do_zero_compiler()
        end subroutine case_test_session_data_implied_do_zero_compiler
    end interface
end module ffc_case_test_session_data_implied_do_zero_compiler

subroutine case_test_session_data_implied_do_zero_compiler()
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none
    save

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

    ! integer(1) tail zeroing exercises the 1-byte element stride in the
    ! memset size (the i8/i16 claim needs a compilable differential case).
    if (.not. expect_output_matches_gfortran( &
        'program p'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    integer(1) :: b(4)'//new_line('a')// &
        '    data b(1) / 5 /'//new_line('a')// &
        '    print *, b'//new_line('a')// &
        'end program p', &
        'data_partial_zero_i8')) stop 1

    ! A statement-label body lowers through the GOTO-aware path, which
    ! bypasses the structured body walk's DATA flush; the labeled path must
    ! flush DATA (with whole-array zero-fill) once before its executables.
    if (.not. expect_output_matches_gfortran( &
        'program p'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    integer :: a(4), i'//new_line('a')// &
        '    data a(1) / 9 /'//new_line('a')// &
        '    i = 1'//new_line('a')// &
        '20  continue'//new_line('a')// &
        '    print *, a'//new_line('a')// &
        '    if (i < 2) then'//new_line('a')// &
        '        i = i + 1'//new_line('a')// &
        '        goto 20'//new_line('a')// &
        '    end if'//new_line('a')// &
        'end program p', &
        'data_partial_zero_labeled')) stop 1

    print *, 'PASS: DATA partial init zero-fills omitted elements'
end subroutine case_test_session_data_implied_do_zero_compiler
