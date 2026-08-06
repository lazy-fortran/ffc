program test_session_forall_alias_compiler
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    all_passed = .true.
    if (.not. test_reverse_direction()) all_passed = .false.
    if (.not. test_ascending_direction()) all_passed = .false.
    if (.not. test_statement_order()) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: FORALL alias assignment preserves pre-assignment values'

contains

    logical function test_reverse_direction()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  integer :: b(5)'//new_line('a')// &
            '  b = [1, 2, 3, 4, 5]'//new_line('a')// &
            '  forall (i = 2:5) b(i) = b(i - 1)'//new_line('a')// &
            '  print *, b'//new_line('a')// &
            'end program main'

        test_reverse_direction = matches_gfortran(source, 'reverse')
    end function test_reverse_direction

    logical function test_ascending_direction()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  integer :: b(5)'//new_line('a')// &
            '  b = [1, 2, 3, 4, 5]'//new_line('a')// &
            '  forall (i = 1:4) b(i) = b(i + 1)'//new_line('a')// &
            '  print *, b'//new_line('a')// &
            'end program main'

        test_ascending_direction = matches_gfortran(source, 'ascending')
    end function test_ascending_direction

    logical function test_statement_order()
        ! The second statement must see all updates made by the first statement,
        ! rather than observing only the current iteration's store.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            '  integer :: c(4)'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  c = 0'//new_line('a')// &
            '  forall (i = 1:3)'//new_line('a')// &
            '    a(i) = a(i + 1)'//new_line('a')// &
            '    c(i) = a(i + 1)'//new_line('a')// &
            '  end forall'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            '  print *, c'//new_line('a')// &
            'end program main'

        test_statement_order = matches_gfortran(source, 'statement_order')
    end function test_statement_order

    logical function matches_gfortran(source, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        base = '/tmp/ffc_forall_alias_'//trim(stem)
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
            print *, 'FAIL[', trim(stem), ']: FortFront rejected source'
            return
        end if
        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL[', trim(stem), ']: ffc lowering failed: ', &
                trim(error_msg)
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
        call execute_command_line(exe//' > '//ffc_out, exitstat=exit_stat)
        call execute_command_line(ref//' > '//ref_out, exitstat=exit_stat)
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL[', trim(stem), ']: ffc output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
        matches_gfortran = .true.
    end function matches_gfortran

end program test_session_forall_alias_compiler
