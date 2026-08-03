program test_session_f32_call_f64_function_compiler
    ! A real(4) assignment must call a contained real(8) function through the
    ! f64 result ABI before narrowing. The mutually recursive calls make this
    ! an independent regression for same-unit procedure lowering (#448).
    use ffc_test_support, only: compile_to_exe
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  real :: x = 5.0, y'//new_line('a')// &
        '  real :: p = 5.0, q'//new_line('a')// &
        '  y = f(x)'//new_line('a')// &
        "  write (*, '(F8.3)') y"//new_line('a')// &
        '  q = f_real(p)'//new_line('a')// &
        "  write (*, '(F8.3)') q"//new_line('a')// &
        'contains'//new_line('a')// &
        '  double precision function f(a) result(b)'//new_line('a')// &
        '    real, intent(in) :: a'//new_line('a')// &
        '    real :: x'//new_line('a')// &
        '    x = 2.0'//new_line('a')// &
        '    b = a + f_real(0.0)'//new_line('a')// &
        '  end function f'//new_line('a')// &
        '  double precision function f_real(a) result(b)'//new_line('a')// &
        '    real, intent(in) :: a'//new_line('a')// &
        '    if (a == 0.0) then'//new_line('a')// &
        '      b = 2.0d0'//new_line('a')// &
        '    else'//new_line('a')// &
        '      b = a + f(1.0)'//new_line('a')// &
        '    end if'//new_line('a')// &
        '  end function f_real'//new_line('a')// &
        'end program main'

    print *, '=== f32 call to contained f64 function compiler test ==='
    if (.not. matches_gfortran(source, '/tmp/ffc_session_f32_f64_call')) stop 1
    print *, 'PASS: f32 calls to contained f64 functions match gfortran'

contains

    logical function matches_gfortran(program_source, stem)
        character(len=*), intent(in) :: program_source
        character(len=*), intent(in) :: stem
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: source_path, ffc_output, ref_output
        character(len=:), allocatable :: ref_exe, ffc_exe
        integer :: unit, exit_stat, cmd_stat, diff_status

        matches_gfortran = .false.
        source_path = trim(stem)//'.f90'
        ffc_exe = trim(stem)//'.ffc'
        ref_exe = trim(stem)//'.gfortran'
        ffc_output = trim(stem)//'.ffc.out'
        ref_output = trim(stem)//'.gfortran.out'

        call compile_to_exe(program_source, ffc_exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=source_path, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//source_path//' -o '//ref_exe, &
                                  exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected the regression source'
            call execute_command_line('rm -f '//source_path//' '//ffc_exe//' '// &
                                      ref_exe//' '//ffc_output//' '//ref_output)
            return
        end if

        call execute_command_line(ffc_exe//' > '//ffc_output, exitstat=exit_stat, &
                                  cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: ffc regression executable failed'
            call execute_command_line('rm -f '//source_path//' '//ffc_exe//' '// &
                                      ref_exe//' '//ffc_output//' '//ref_output)
            return
        end if
        call execute_command_line(ref_exe//' > '//ref_output, exitstat=exit_stat, &
                                  cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: gfortran regression executable failed'
            call execute_command_line('rm -f '//source_path//' '//ffc_exe//' '// &
                                      ref_exe//' '//ffc_output//' '//ref_output)
            return
        end if
        call execute_command_line('diff -u '//ffc_output//' '//ref_output, &
                                  exitstat=diff_status, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. diff_status /= 0) then
            print *, 'FAIL: ffc output differs from gfortran'
            call execute_command_line('rm -f '//source_path//' '//ffc_exe//' '// &
                                      ref_exe//' '//ffc_output//' '//ref_output)
            return
        end if

        call execute_command_line('rm -f '//source_path//' '//ffc_exe//' '// &
                                  ref_exe//' '//ffc_output//' '//ref_output)
        matches_gfortran = .true.

    end function matches_gfortran

end program test_session_f32_call_f64_function_compiler
