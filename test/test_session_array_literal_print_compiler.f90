program test_session_array_literal_print_compiler
    ! Verify list-directed print of an inline array constructor [e1, e2, ...]
    ! lowers through the direct LIRIC session and prints byte-for-byte like
    ! gfortran. Each element (literal, variable, or non-contained function call)
    ! is emitted like a scalar print item.
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session array-literal print test ==='

    all_passed = .true.

    ! integer literal constructor
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  print *, [4, 9, 16]'//new_line('a')// &
        'end program main', &
        'int_literals')) all_passed = .false.

    ! real literal constructor
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  print *, [1.0, 2.5, 3.0]'//new_line('a')// &
        'end program main', &
        'real_literals')) all_passed = .false.

    ! variable and expression elements
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer :: a'//new_line('a')// &
        '  a = 5'//new_line('a')// &
        '  print *, [a, a + 1, a * 2]'//new_line('a')// &
        'end program main', &
        'var_elements')) all_passed = .false.

    ! module-function calls as constructor elements
    if (.not. matches_gfortran( &
        'module m'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        'contains'//new_line('a')// &
        '  integer function sq(x)'//new_line('a')// &
        '    integer, intent(in) :: x'//new_line('a')// &
        '    sq = x * x'//new_line('a')// &
        '  end function sq'//new_line('a')// &
        'end module m'//new_line('a')// &
        'program main'//new_line('a')// &
        '  use m'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  print *, [sq(1), sq(2), sq(3)]'//new_line('a')// &
        'end program main', &
        'fn_call_elements')) all_passed = .false.

    ! array literal among other list-directed items
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  print *, "vals:", [10, 20, 30]'//new_line('a')// &
        'end program main', &
        'mixed_items')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: inline array-constructor print lowers through '// &
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
        base = '/tmp/ffc_arrlit_'//trim(stem)
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

end program test_session_array_literal_print_compiler
