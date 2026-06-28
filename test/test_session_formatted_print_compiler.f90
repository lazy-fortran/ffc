program test_session_formatted_print_compiler
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, INPUT_MODE_STANDARD
    use ffc_test_support, only: expect_output
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session formatted print compiler test ==='

    all_passed = .true.
    if (.not. test_integer_i0()) all_passed = .false.
    if (.not. test_integer_i5()) all_passed = .false.
    if (.not. test_string_a_literal()) all_passed = .false.
    if (.not. test_string_a_variable()) all_passed = .false.
    if (.not. test_compound_i_x_f()) all_passed = .false.
    if (.not. test_compound_a_i()) all_passed = .false.
    if (.not. test_compound_a_width()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: formatted print lowers through direct LIRIC session'

contains

    logical function test_integer_i0()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  i = 42'//new_line('a')// &
            "  print '(I0)', i"//new_line('a')// &
            'end program main'

        test_integer_i0 = expect_output(source, '42'//new_line('a'), &
            '/tmp/ffc_fmt_i0_test')
    end function test_integer_i0

    logical function test_integer_i5()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  i = 42'//new_line('a')// &
            "  print '(I5)', i"//new_line('a')// &
            'end program main'

        test_integer_i5 = expect_output(source, '   42'//new_line('a'), &
            '/tmp/ffc_fmt_i5_test')
    end function test_integer_i5

    logical function test_string_a_literal()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            "  print '(A)', 'hello'"//new_line('a')// &
            'end program main'

        test_string_a_literal = expect_output(source, 'hello'//new_line('a'), &
            '/tmp/ffc_fmt_a_lit_test')
    end function test_string_a_literal

    logical function test_string_a_variable()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: s'//new_line('a')// &
            "  s = 'hello'"//new_line('a')// &
            "  print '(A)', s"//new_line('a')// &
            'end program main'

        test_string_a_variable = matches_gfortran(source, 'fmt_a_var')
    end function test_string_a_variable

    logical function test_compound_i_x_f()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  real :: x'//new_line('a')// &
            '  i = 7'//new_line('a')// &
            '  x = 3.25'//new_line('a')// &
            "  print '(I5,2X,F8.3)', i, x"//new_line('a')// &
            'end program main'

        test_compound_i_x_f = matches_gfortran(source, 'fmt_compound_i_x_f')
    end function test_compound_i_x_f

    logical function test_compound_a_i()
        ! Compound format mixing A (character) and I descriptors with literal
        ! text and a variable, matched byte-for-byte against gfortran.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  character(len=5) :: s'//new_line('a')// &
            "  s = 'count'"//new_line('a')// &
            '  n = 42'//new_line('a')// &
            "  print '(A,A,I5)', s, ' = ', n"//new_line('a')// &
            'end program main'

        test_compound_a_i = matches_gfortran(source, 'fmt_compound_a_i')
    end function test_compound_a_i

    logical function test_compound_a_width()
        ! Aw (width) on a character literal inside a compound format.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  i = 7'//new_line('a')// &
            "  print '(A8,I3)', 'tag', i"//new_line('a')// &
            'end program main'

        test_compound_a_width = matches_gfortran(source, 'fmt_compound_a_width')
    end function test_compound_a_width

    logical function matches_gfortran(source, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        base = '/tmp/ffc_'//stem
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
            print *, 'FAIL[', stem, ']: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, &
            error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL[', stem, ']: ffc lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', stem, ']: gfortran rejected source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out, exitstat=exit_stat)
        call execute_command_line(ref//' > '//ref_out, exitstat=exit_stat)
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL[', stem, ']: ffc output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
        else
            matches_gfortran = .true.
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
    end function matches_gfortran

end program test_session_formatted_print_compiler
