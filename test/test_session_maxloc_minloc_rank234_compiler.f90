program test_session_maxloc_minloc_rank234_compiler
    use fortfront, only: compile_frontend_from_string, &
        compiler_frontend_options_t, compiler_frontend_result_t
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed
    character(len=*), parameter :: rank2_source = &
        'program main'//new_line('a')// &
        '  integer :: a(2,3), hi(2), lo(2); logical :: m(2,3)'//new_line('a')// &
        '  a = reshape([4, 9, 9, 2, 7, 1], [2, 3])'//new_line('a')// &
        '  m = reshape([.true., .false., .false., .true., .true., .false.], [2, 3])'//new_line('a')// &
        '  hi = maxloc(a)'//new_line('a')// &
        '  lo = minloc(a)'//new_line('a')// &
        '  if (any(hi /= [2, 1]) .or. any(lo /= [2, 3])) error stop 1'//new_line('a')// &
        '  hi = maxloc(a, mask=m)'//new_line('a')// &
        '  lo = minloc(a, mask=m)'//new_line('a')// &
        '  if (any(hi /= [1, 3]) .or. any(lo /= [2, 2])) error stop 2'//new_line('a')// &
        '  print *, hi, lo'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank3_source = &
        'program main'//new_line('a')// &
        '  integer :: a(2,2,2), hi(3), lo(3); logical :: m(2,2,2)'//new_line('a')// &
        '  a = reshape([5, 1, 8, 8, 2, 7, 3, 7], [2, 2, 2])'//new_line('a')// &
        '  m = reshape([.false., .true., .false., .true., .false., .true., .false., .false.], [2, 2, 2])'//new_line('a')// &
        '  hi = maxloc(a)'//new_line('a')// &
        '  lo = minloc(a)'//new_line('a')// &
        '  if (any(hi /= [1, 2, 1]) .or. any(lo /= [2, 1, 1])) error stop 3'//new_line('a')// &
        '  hi = maxloc(a, mask=m)'//new_line('a')// &
        '  lo = minloc(a, mask=m)'//new_line('a')// &
        '  if (any(hi /= [2, 2, 1]) .or. any(lo /= [2, 1, 1])) error stop 4'//new_line('a')// &
        '  print *, hi, lo'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: rank4_source = &
        'program main'//new_line('a')// &
        '  integer :: a(2,2,2,2), hi(4), lo(4); logical :: m(2,2,2,2)'//new_line('a')// &
        '  a = 0'//new_line('a')// &
        '  a(1,1,1,1) = 1; a(2,1,1,1) = 2; a(1,2,1,1) = 3; a(2,2,1,1) = 4'//new_line('a')// &
        '  a(1,1,2,1) = 9; a(2,1,2,1) = 6; a(1,2,2,1) = 0; a(2,2,2,1) = 8'//new_line('a')// &
        '  a(1,1,1,2) = 5; a(2,1,1,2) = 10; a(1,2,1,2) = 11; a(2,2,1,2) = 9'//new_line('a')// &
        '  a(1,1,2,2) = 13; a(2,1,2,2) = 14; a(1,2,2,2) = 15; a(2,2,2,2) = 16'//new_line('a')// &
        '  m = .false.; m(1,2,2,1) = .true.; m(1,2,1,2) = .true.; m(2,2,1,2) = .true.'//new_line('a')// &
        '  hi = maxloc(a)'//new_line('a')// &
        '  lo = minloc(a)'//new_line('a')// &
        '  if (any(hi /= [2, 2, 2, 2]) .or. any(lo /= [1, 2, 2, 1])) error stop 5'//new_line('a')// &
        '  hi = maxloc(a, mask=m)'//new_line('a')// &
        '  lo = minloc(a, mask=m)'//new_line('a')// &
        '  if (any(hi /= [1, 2, 1, 2]) .or. any(lo /= [1, 2, 2, 1])) error stop 6'//new_line('a')// &
        '  print *, hi, lo'//new_line('a')// &
        'end program main'

    print *, '=== direct session rank-2/3/4 MAXLOC/MINLOC test ==='
    all_passed = matches_gfortran(rank2_source, &
        '/var/tmp/ert/ffc_maxloc_minloc_rank2')
    all_passed = matches_gfortran(rank3_source, &
        '/var/tmp/ert/ffc_maxloc_minloc_rank3') .and. all_passed
    all_passed = matches_gfortran(rank4_source, &
        '/var/tmp/ert/ffc_maxloc_minloc_rank4') .and. all_passed
    if (.not. all_passed) stop 1
    print *, 'PASS: rank-2/3/4 coordinate vectors match gfortran'

contains

    logical function matches_gfortran(program_source, base)
        character(len=*), intent(in) :: program_source, base
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, diff_status

        matches_gfortran = .false.
        src = trim(base)//'.f90'
        exe = trim(base)//'.ffc'
        ref = trim(base)//'.gfortran'
        ffc_out = trim(base)//'.ffc.out'
        ref_out = trim(base)//'.gfortran.out'
        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        call compile_frontend_from_string(program_source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected rank-specific source'
            print *, trim(frontend_result%diagnostic_text)
            return
        end if
        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc lowering failed: ', trim(error_msg)
            return
        end if
        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected rank-specific source'
            return
        end if
        call execute_command_line(exe//' > '//ffc_out//' 2>&1', &
            exitstat=exit_stat)
        call execute_command_line(ref//' > '//ref_out//' 2>&1', &
            exitstat=exit_stat)
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=diff_status)
        if (diff_status /= 0) then
            print *, 'FAIL: ffc output differs from gfortran for '//trim(base)
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
        matches_gfortran = .true.
    end function matches_gfortran

end program test_session_maxloc_minloc_rank234_compiler
