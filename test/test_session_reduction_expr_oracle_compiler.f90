program test_session_reduction_expr_oracle_compiler
    ! Differential oracle for an allocatable-array function result used as a
    ! reduction argument.  The function mutates host state, so evaluating the
    ! reduction expression more than once changes the observable result.
    use ffc_test_support, only: compile_to_exe
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  integer :: calls, total'//new_line('a')// &
        '  calls = 0'//new_line('a')// &
        '  total = sum(make())'//new_line('a')// &
        '  print *, calls, total'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function make() result(v)'//new_line('a')// &
        '    integer, allocatable :: v(:)'//new_line('a')// &
        '    calls = calls + 1'//new_line('a')// &
        '    allocate(v(3))'//new_line('a')// &
        '    v = [1, 2, 3]'//new_line('a')// &
        '  end function make'//new_line('a')// &
        'end program main'

    print *, '=== reduction expression gfortran oracle test ==='
    if (.not. matches_gfortran(source)) stop 1
    print *, 'PASS: allocatable reduction argument evaluates exactly once'

contains

    logical function matches_gfortran(program_source)
        character(len=*), intent(in) :: program_source
        character(len=:), allocatable :: error_msg, source_path, ffc_exe
        character(len=:), allocatable :: gfortran_exe, ffc_out, gfortran_out
        integer :: unit, ffc_status, gfortran_status, command_status

        matches_gfortran = .false.
        source_path = '/tmp/ffc_reduction_expr_oracle.f90'
        ffc_exe = '/tmp/ffc_reduction_expr_oracle.ffc'
        gfortran_exe = '/tmp/ffc_reduction_expr_oracle.gfortran'
        ffc_out = '/tmp/ffc_reduction_expr_oracle.ffc.out'
        gfortran_out = '/tmp/ffc_reduction_expr_oracle.gfortran.out'

        call compile_to_exe(program_source, ffc_exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc lowering failed: ', trim(error_msg)
            call cleanup()
            return
        end if

        open (newunit=unit, file=source_path, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//source_path//' -o '// &
            gfortran_exe, exitstat=gfortran_status, &
            cmdstat=command_status)
        if (command_status /= 0 .or. gfortran_status /= 0) then
            print *, 'FAIL: gfortran rejected the regression source'
            call cleanup()
            return
        end if

        call execute_command_line('timeout 5s '//ffc_exe//' > '//ffc_out, &
            exitstat=ffc_status, cmdstat=command_status)
        if (command_status /= 0 .or. ffc_status /= 0) then
            print *, 'FAIL: ffc regression executable could not run'
            call cleanup()
            return
        end if
        call execute_command_line('timeout 5s '//gfortran_exe//' > '// &
            gfortran_out, exitstat=gfortran_status, &
            cmdstat=command_status)
        if (command_status /= 0 .or. gfortran_status /= 0) then
            print *, 'FAIL: gfortran regression executable could not run'
            call cleanup()
            return
        end if
        call execute_command_line('diff -u '//gfortran_out//' '//ffc_out, &
            exitstat=command_status)
        if (command_status /= 0) then
            print *, 'FAIL: ffc output differs from gfortran'
            call cleanup()
            return
        end if
        call cleanup()
        matches_gfortran = .true.
    end function matches_gfortran

    subroutine cleanup()
        call execute_command_line('rm -f /tmp/ffc_reduction_expr_oracle.*')
    end subroutine cleanup

end program test_session_reduction_expr_oracle_compiler
