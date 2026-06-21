program test_session_array_function_result_compiler
    ! Verify contained functions returning a fixed-size rank-1 array lower
    ! through the sret result ABI: the caller passes the destination buffer as
    ! the hidden result pointer, and assignment or print of the call matches
    ! gfortran byte-for-byte.
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
                                   compiler_frontend_result_t, &
                                   compile_frontend_from_string, &
                                   INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session array function result test ==='

    all_passed = .true.

    ! Real array result assigned, then printed.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real, dimension(3) :: r'//new_line('a')// &
        '  r = make_vec()'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function make_vec() result(out)'//new_line('a')// &
        '    real, dimension(3) :: out'//new_line('a')// &
        '    out = [1.0, 2.0, 3.0]'//new_line('a')// &
        '  end function make_vec'//new_line('a')// &
        'end program main', &
        'real_assign')) all_passed = .false.

    ! Real array result printed directly from the call.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  print *, make_vec()'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function make_vec() result(out)'//new_line('a')// &
        '    real, dimension(3) :: out'//new_line('a')// &
        '    out = [1.0, 2.0, 3.0]'//new_line('a')// &
        '  end function make_vec'//new_line('a')// &
        'end program main', &
        'real_print')) all_passed = .false.

    ! Array result that reads its array argument (sret + reference args).
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real, dimension(3) :: x, y'//new_line('a')// &
        '  x = [1.0, 2.0, 3.0]'//new_line('a')// &
        '  y = double_array(x)'//new_line('a')// &
        '  print *, y'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function double_array(arr) result(out)'//new_line('a')// &
        '    real, dimension(3), intent(in) :: arr'//new_line('a')// &
        '    real, dimension(3) :: out'//new_line('a')// &
        '    out = arr * 2.0'//new_line('a')// &
        '  end function double_array'//new_line('a')// &
        'end program main', &
        'real_arg')) all_passed = .false.

    ! Integer array result.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer, dimension(4) :: r'//new_line('a')// &
        '  r = ints()'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function ints() result(out)'//new_line('a')// &
        '    integer, dimension(4) :: out'//new_line('a')// &
        '    out = [10, 20, 30, 40]'//new_line('a')// &
        '  end function ints'//new_line('a')// &
        'end program main', &
        'int_assign')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: fixed-size array function results lower through '// &
        'direct LIRIC session'

contains

    logical function matches_gfortran(source, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        base = '/tmp/ffc_arrfn_'//trim(stem)
        src = base//'.f90'
        exe = base//'.ffc'
        ref = base//'.gf'
        ffc_out = base//'.ffc.out'
        ref_out = base//'.gf.out'

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL[', trim(stem), ']: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
                                        frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL[', trim(stem), ']: ffc lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
                                  exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', trim(stem), ']: gfortran rejected source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out//' 2>&1', exitstat=exit_stat)
        call execute_command_line(ref//' > '//ref_out//' 2>&1', exitstat=exit_stat)
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
                                  ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL[', trim(stem), ']: ffc output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref// &
                                  ' '//ffc_out//' '//ref_out)
        matches_gfortran = .true.
    end function matches_gfortran

end program test_session_array_function_result_compiler
