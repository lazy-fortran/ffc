program test_session_runtime_rank4_array_compiler
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  call work(4, 3, 2, 2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(l, m, n, o)'//new_line('a')// &
        '    integer, intent(in) :: l, m, n, o'//new_line('a')// &
        '    integer :: a(l, m, n), b(l, m, n, o)'//new_line('a')// &
        '    a = 1'//new_line('a')// &
        '    b = 2'//new_line('a')// &
        '    print *, a(1, 1, 1), a(l, m, n)'//new_line('a')// &
        '    print *, b(1, 1, 1, 1), b(l, m, n, o)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'

    print *, '=== runtime rank-four automatic array compiler test ==='
    if (.not. matches_gfortran(source, 'runtime_rank4')) stop 1
    print *, 'PASS: runtime rank-four automatic arrays match gfortran'

contains

    logical function matches_gfortran(program_source, stem)
        character(len=*), intent(in) :: program_source, stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        base = '/tmp/ffc_runtime_rank4_'//stem
        src = base//'.f90'
        exe = base//'.ffc'
        ref = base//'.gf'
        ffc_out = base//'.ffc.out'
        ref_out = base//'.gf.out'

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(program_source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL[', stem, ']: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL[', stem, ']: ffc lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
                                  exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', stem, ']: gfortran rejected source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out, exitstat=exit_stat)
        call execute_command_line(ref//' > '//ref_out, exitstat=exit_stat)
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
                                  ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL[', stem, ']: ffc output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
        else
            matches_gfortran = .true.
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
                                  ffc_out//' '//ref_out)
    end function matches_gfortran

end program test_session_runtime_rank4_array_compiler
