program test_session_nested_character_substring_compiler
    ! #669: a substring of a character-array element is a scalar character
    ! view.  Check the four operations that previously lost the nested base:
    ! read, literal write, overlapping assignment, and an assumed-length
    ! actual argument.  gfortran is the independent behavioral oracle.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== nested character-array substring compiler test ==='
    all_passed = matches_gfortran( &
        'program main'//new_line('a')// &
        '  character(len=6) :: c(2)'//new_line('a')// &
        '  character(len=3) :: first'//new_line('a')// &
        '  logical :: ok'//new_line('a')// &
        '  c(1) = "abcdef"'//new_line('a')// &
        '  c(2) = "ghijkl"'//new_line('a')// &
        '  first = c(2)(1:3)'//new_line('a')// &
        '  ok = first == "ghi"'//new_line('a')// &
        '  c(2)(1:3) = "XYZ"'//new_line('a')// &
        '  ok = ok .and. c(2) == "XYZjkl"'//new_line('a')// &
        '  c(2) = "ghijkl"'//new_line('a')// &
        '  c(2)(1:3) = c(2)(2:4)'//new_line('a')// &
        '  ok = ok .and. c(2) == "hijjkl"'//new_line('a')// &
        '  call consume(c(2)(2:5), ok)'//new_line('a')// &
        '  if (ok) then'//new_line('a')// &
        '    print *, "PASS: nested character-array substrings"'//new_line('a')// &
        '  else'//new_line('a')// &
        '    print *, "FAIL: nested character-array substrings"'//new_line('a')// &
        '    stop 1'//new_line('a')// &
        '  end if'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine consume(value, ok)'//new_line('a')// &
        '    character(len=*), intent(in) :: value'//new_line('a')// &
        '    logical, intent(inout) :: ok'//new_line('a')// &
        '    ok = ok .and. value == "ijjk"'//new_line('a')// &
        '  end subroutine consume'//new_line('a')// &
        'end program main', 'nested_char_substring')

    if (.not. all_passed) stop 1
    print *, 'PASS: nested character-array substrings match gfortran'

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
        base = '/var/tmp/ert/ffc_issue669_'//trim(stem)
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

        call execute_command_line(exe//' > '//ffc_out//' 2>&1', &
            exitstat=exit_stat)
        call execute_command_line(ref//' > '//ref_out//' 2>&1', &
            exitstat=exit_stat)
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

end program test_session_nested_character_substring_compiler
