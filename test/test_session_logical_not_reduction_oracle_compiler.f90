program test_session_logical_not_reduction_oracle_compiler
    ! Keep this fixture independent from the broader whole-array regression:
    ! the FFC API and gfortran must agree on ANY(.NOT. array).
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  logical :: a(3)'//new_line('a')// &
        '  a = [.true., .false., .true.]'//new_line('a')// &
        '  print *, any(.not. a), any(a)'//new_line('a')// &
        'end program main'

    print *, '=== logical NOT reduction compiler/API oracle test ==='
    if (.not. matches_gfortran()) stop 1
    print *, 'PASS: ANY(.NOT. array) matches gfortran'

contains

    logical function matches_gfortran() result(ok)
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=*), parameter :: base = &
            '/var/tmp/ert/ffc_logical_not_reduction_oracle'
        character(len=*), parameter :: src = base//'.f90'
        character(len=*), parameter :: ffc_exe = base//'.ffc'
        character(len=*), parameter :: gfortran_exe = base//'.gfortran'
        character(len=*), parameter :: ffc_out = base//'.ffc.out'
        character(len=*), parameter :: gfortran_out = base//'.gfortran.out'
        integer :: unit, exit_stat, diff_status

        ok = .false.
        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected oracle source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, ffc_exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: FFC logical NOT reduction lowering failed: ', &
                trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//gfortran_exe, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected oracle source'
            return
        end if

        call execute_command_line(ffc_exe//' > '//ffc_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: FFC oracle executable failed'
            return
        end if
        call execute_command_line(gfortran_exe//' > '//gfortran_out, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran oracle executable failed'
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//gfortran_out// &
            ' > /dev/null 2>&1', exitstat=diff_status)
        if (diff_status /= 0) then
            print *, 'FAIL: FFC oracle output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//gfortran_out)
            return
        end if

        call execute_command_line('rm -f '//src//' '//ffc_exe//' '// &
            gfortran_exe//' '//ffc_out//' '//gfortran_out)
        ok = .true.
    end function matches_gfortran

end program test_session_logical_not_reduction_oracle_compiler
