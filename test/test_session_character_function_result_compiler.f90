program test_session_character_function_result_compiler
    ! Verify character-returning contained functions lower correctly: a
    ! fixed-length result (character(len=N) :: s), a runtime-length result
    ! (character(len=k) where k is a dummy argument), and a character variable
    ! declared with a parameter length (character(max_len)). Each program's
    ! output must match gfortran byte-for-byte (#1614).
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session character function result compiler test ==='

    all_passed = .true.

    ! Fixed-length character result printed by the caller.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  character(len=5) :: r'//new_line('a')// &
        '  r = get_name()'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function get_name() result(s)'//new_line('a')// &
        '    character(len=5) :: s'//new_line('a')// &
        '    s = "Hello"'//new_line('a')// &
        '  end function get_name'//new_line('a')// &
        'end program main', &
        'fixed_result')) all_passed = .false.

    ! Runtime-length character result (length is a dummy argument).
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  character(len=:), allocatable :: r'//new_line('a')// &
        '  r = make(4)'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function make(k) result(s)'//new_line('a')// &
        '    integer, intent(in) :: k'//new_line('a')// &
        '    character(len=k) :: s'//new_line('a')// &
        '    s = repeat("Z", k)'//new_line('a')// &
        '  end function make'//new_line('a')// &
        'end program main', &
        'runtime_result')) all_passed = .false.

    ! Character variable declared with a parameter length.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer, parameter :: max_len = 6'//new_line('a')// &
        '  character(max_len) :: name'//new_line('a')// &
        '  name = "Test"'//new_line('a')// &
        '  print *, name'//new_line('a')// &
        'end program main', &
        'param_length')) all_passed = .false.

    ! Result length taken from a dummy via len(): character(len=len(name)).
    ! The actual width comes from the assigned value, so the deferred ABI
    ! resolves it without evaluating the declared length expression (#1407).
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  print *, greet("Ada")'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function greet(name) result(s)'//new_line('a')// &
        '    character(len=*), intent(in) :: name'//new_line('a')// &
        '    character(len=len(name)) :: s'//new_line('a')// &
        '    s = "" // name'//new_line('a')// &
        '  end function greet'//new_line('a')// &
        'end program main', &
        'len_of_dummy')) all_passed = .false.

    ! Result length is an expression over a dummy: character(len=len(name)+7).
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  print *, greet("Ada")'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function greet(name) result(s)'//new_line('a')// &
        '    character(len=*), intent(in) :: name'//new_line('a')// &
        '    character(len=len(name)+7) :: s'//new_line('a')// &
        '    s = "Hello, " // name'//new_line('a')// &
        '  end function greet'//new_line('a')// &
        'end program main', &
        'len_expr_of_dummy')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: character function results lower through direct LIRIC session'

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
        base = '/tmp/ffc_charfn_'//trim(stem)
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

end program test_session_character_function_result_compiler
