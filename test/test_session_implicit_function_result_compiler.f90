program test_session_implicit_function_result_compiler
    ! Verify a function whose result type is left to Fortran implicit typing
    ! (no explicit result type and no body declaration of the result variable)
    ! lowers with the kind derived from the default letter rule: i-n integer,
    ! otherwise real. The result must print byte-for-byte like gfortran.
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session implicit function result test ==='

    all_passed = .true.

    ! result name starts with n: implicit integer
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer :: r'//new_line('a')// &
        '  r = nfun(3)'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function nfun(x)'//new_line('a')// &
        '    integer :: x'//new_line('a')// &
        '    nfun = x + 10'//new_line('a')// &
        '  end function nfun'//new_line('a')// &
        'end program main', &
        'implicit_integer')) all_passed = .false.

    ! result name starts with a: implicit real
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real :: r'//new_line('a')// &
        '  r = afun(2.0)'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function afun(x)'//new_line('a')// &
        '    real :: x'//new_line('a')// &
        '    afun = x * 1.5'//new_line('a')// &
        '  end function afun'//new_line('a')// &
        'end program main', &
        'implicit_real')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: implicitly-typed function results lower through '// &
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
        base = '/tmp/ffc_implres_'//trim(stem)
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

end program test_session_implicit_function_result_compiler
