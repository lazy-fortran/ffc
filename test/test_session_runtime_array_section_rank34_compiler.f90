program test_session_runtime_array_section_rank34_compiler
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  call work(4, 3, 2, 2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(n1, n2, n3, n4)'//new_line('a')// &
        '    integer, intent(in) :: n1, n2, n3, n4'//new_line('a')// &
        '    integer :: a(n1,n2,n3), b(n1,n2,n3,n4)'//new_line('a')// &
        '    integer :: i, j, k, l, i0, j0, k0, l0'//new_line('a')// &
        '    a = 1'//new_line('a')// &
        '    b = 2'//new_line('a')// &
        '    i0 = 2'//new_line('a')// &
        '    j0 = 2'//new_line('a')// &
        '    k0 = 2'//new_line('a')// &
        '    l0 = 2'//new_line('a')// &
        '    a(i0:n1-1,j0:n2-1,k0:n3) = 11'//new_line('a')// &
        '    b(i0:n1-1,j0:n2-1,k0:n3,l0:n4) = 22'//new_line('a')// &
        '    if (a(1,1,1) /= 1) stop 11'//new_line('a')// &
        '    if (a(2,2,2) /= 11) stop 12'//new_line('a')// &
        '    if (a(4,3,2) /= 1) stop 13'//new_line('a')// &
        '    if (b(1,1,1,1) /= 2) stop 14'//new_line('a')// &
        '    if (b(2,2,2,2) /= 22) stop 15'//new_line('a')// &
        '    if (b(4,3,2,1) /= 2) stop 16'//new_line('a')// &
        '    print *, a(1,1,1), a(2,1,1), a(2,2,2), a(4,3,2)'//new_line('a')// &
        '    print *, b(1,1,1,1), b(2,1,1,1), b(2,2,2,2), b(4,3,2,1)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    print *, '=== direct session runtime rank-3/rank-4 section test ==='
    if (.not. matches_gfortran(source)) stop 1
    print *, 'PASS: runtime rank-3 and rank-4 sections match independent checks and gfortran'

contains

    logical function matches_gfortran(program_source) result(ok)
        character(len=*), intent(in) :: program_source
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, diff_status

        ok = .false.
        base = '/var/tmp/ert/ffc_runtime_section_rank34'
        src = base//'.f90'
        exe = base//'.ffc'
        ref = base//'.gfortran'
        ffc_out = base//'.ffc.out'
        ref_out = base//'.gfortran.out'

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(program_source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected rank-3/rank-4 runtime sections: ', &
                trim(frontend_result%diagnostic_text)
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
            print *, 'FAIL: gfortran rejected rank-3/rank-4 runtime sections'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: ffc rank-3/rank-4 runtime section executable failed'
            return
        end if
        call execute_command_line(ref//' > '//ref_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rank-3/rank-4 runtime section executable failed'
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=diff_status)
        if (diff_status /= 0) then
            print *, 'FAIL: rank-3/rank-4 runtime section output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if

        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
        ok = .true.
    end function matches_gfortran

end program test_session_runtime_array_section_rank34_compiler
