module diagnostic_gfortran_oracle
    use ffc_test_support, only: compile_to_exe
    implicit none
    private
    public :: matches_gfortran

contains

    logical function matches_gfortran(source, stem, cli) result(ok)
        character(len=*), intent(in) :: source, stem
        logical, intent(in) :: cli
        character(len=:), allocatable :: base, source_path, exe, ref, error_msg
        integer :: unit, status

        ok = .false.
        base = '/var/tmp/ffc_diagnostic_oracle_'//trim(stem)
        source_path = base//'.f90'
        exe = base//'.ffc'
        ref = base//'.gfortran'
        if (.not. run_checked('rm -f '//exe//' '//ref, &
            stem//' remove stale executables')) return
        open (newunit=unit, file=source_path, status='replace', action='write', &
            iostat=status)
        if (status /= 0) then
            print *, 'FAIL[', stem, ']: cannot write oracle source'
            return
        end if
        write (unit, '(A)') source
        close (unit)
        if (cli) then
            if (.not. run_checked('fo exec --no-build ffc '//source_path// &
                ' -o '//exe//' > '//base//'.compile.log 2>&1', &
                stem//' CLI compile')) return
        else
            call compile_to_exe(source, exe, error_msg)
            if (len_trim(error_msg) > 0) then
                print *, 'FAIL[', stem, ']: ', error_msg
                return
            end if
        end if
        ! GNU mode is intentional: main-program RETURN and real subscripts
        ! are supported extensions whose behavior needs an independent oracle.
        if (.not. run_checked('gfortran -w '//source_path//' -o '//ref// &
            ' > '//base//'.reference.log 2>&1', &
            stem//' reference compile')) return
        if (.not. run_checked(exe//' > '//base//'.ffc.out 2>&1', &
            stem//' ffc execution')) return
        if (.not. run_checked(ref//' > '//base//'.gfortran.out 2>&1', &
            stem//' reference execution')) return
        ok = run_checked('cmp '//base//'.ffc.out '//base//'.gfortran.out', &
            stem//' output comparison')
    end function matches_gfortran

    logical function run_checked(command, label) result(ok)
        character(len=*), intent(in) :: command, label
        integer :: cmd_status, exit_status

        call execute_command_line(command, cmdstat=cmd_status, exitstat=exit_status)
        ok = .false.
        if (cmd_status /= 0) then
            print *, 'FAIL[', label, ']: cannot execute command'
            return
        end if
        if (exit_status /= 0) then
            print *, 'FAIL[', label, ']: exit status ', exit_status
            return
        end if
        ok = .true.
    end function run_checked

end module diagnostic_gfortran_oracle
