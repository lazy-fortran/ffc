program test_session_fixed_char_function_result_compiler
    ! #410: a fixed-length CHARACTER function result (character(len=N) :: s)
    ! returns exactly its declared length N. A shorter value pads with blanks,
    ! a longer one truncates, embedded blanks survive, and LEN of the result is
    ! N. Each program's LEN and printed bytes must match gfortran. The negative
    ! case checks that a nonconstant result length taken from a local variable
    ! is rejected as an invalid specification expression.
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session fixed-length character result compiler test ==='

    all_passed = .true.

    ! A value shorter than the declared length pads with blanks; LEN is N.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  character(len=8) :: r'//new_line('a')// &
        '  r = short()'//new_line('a')// &
        '  print *, len(r), "[", r, "]"'//new_line('a')// &
        '  print *, len(short()), "[", short(), "]"'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function short() result(s)'//new_line('a')// &
        '    character(len=8) :: s'//new_line('a')// &
        '    s = "Hi"'//new_line('a')// &
        '  end function short'//new_line('a')// &
        'end program main', &
        'pad_short')) all_passed = .false.

    ! A value of exactly the declared length is returned unchanged.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  print *, len(exact()), "[", exact(), "]"'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function exact() result(s)'//new_line('a')// &
        '    character(len=3) :: s'//new_line('a')// &
        '    s = "abc"'//new_line('a')// &
        '  end function exact'//new_line('a')// &
        'end program main', &
        'exact')) all_passed = .false.

    ! A longer value truncates to the declared length.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  print *, len(trunc()), "[", trunc(), "]"'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function trunc() result(s)'//new_line('a')// &
        '    character(len=4) :: s'//new_line('a')// &
        '    s = "abcdefgh"'//new_line('a')// &
        '  end function trunc'//new_line('a')// &
        'end program main', &
        'truncate')) all_passed = .false.

    ! A concatenated value pads to the declared length and keeps embedded
    ! blanks.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  character(len=12) :: r'//new_line('a')// &
        '  r = embed("a b")'//new_line('a')// &
        '  print *, len(r), "[", r, "]"'//new_line('a')// &
        '  print *, len(embed("a b")), "[", embed("a b"), "]"'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function embed(x) result(s)'//new_line('a')// &
        '    character(len=*), intent(in) :: x'//new_line('a')// &
        '    character(len=12) :: s'//new_line('a')// &
        '    s = x // "  c"'//new_line('a')// &
        '  end function embed'//new_line('a')// &
        'end program main', &
        'concat_pad')) all_passed = .false.

    ! Negative: a result length taken from a local variable is not a valid
    ! specification expression and must be diagnosed, not silently lowered.
    if (.not. rejects_lowering( &
        'program main'//new_line('a')// &
        '  print *, bad()'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function bad() result(s)'//new_line('a')// &
        '    integer :: n'//new_line('a')// &
        '    character(len=n) :: s'//new_line('a')// &
        '    s = "abc"'//new_line('a')// &
        '  end function bad'//new_line('a')// &
        'end program main', &
        'local_length')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: fixed-length character results keep their declared length'

contains

    logical function matches_gfortran(source, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        base = '/tmp/ffc_fixedcharfn_'//trim(stem)
        src = base//'.f90'
        exe = base//'.ffc'
        ref = base//'.gf'
        ffc_out = base//'.ffc.out'
        ref_out = base//'.gf.out'

        if (.not. lower_source(source, stem, exe, frontend_result, error_msg)) &
            return

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

    logical function rejects_lowering(source, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: exe

        rejects_lowering = .false.
        exe = '/tmp/ffc_fixedcharfn_'//trim(stem)//'.ffc'

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            rejects_lowering = .true.
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) == 0) then
            print *, 'FAIL[', trim(stem), ']: invalid result length was accepted'
            return
        end if
        if (index(error_msg, 'specification expression') == 0) then
            print *, 'FAIL[', trim(stem), ']: unexpected diagnostic: ', &
                trim(error_msg)
            return
        end if
        rejects_lowering = .true.
    end function rejects_lowering

    logical function lower_source(source, stem, exe, frontend_result, error_msg)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        character(len=*), intent(in) :: exe
        type(compiler_frontend_result_t), intent(out) :: frontend_result
        character(len=:), allocatable, intent(out) :: error_msg
        type(compiler_frontend_options_t) :: options

        lower_source = .false.
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
            print *, 'FAIL[', trim(stem), ']: ffc lowering failed: ', &
                trim(error_msg)
            return
        end if
        lower_source = .true.
    end function lower_source

end program test_session_fixed_char_function_result_compiler
